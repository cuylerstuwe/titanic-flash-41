//! Optional parking must not spend the parent conversation or a needed prefix.
//! Root-only CPU metadata becomes a bounded mask in the ordered park command.
use super::*;
use std::rc::Rc;
use crate::prompt_media::Identity;

struct Prefix {
    tokens: Vec<u32>,
    media: Identity,
    _reservation: Option<crate::memory::HostWorkspaceReservation>,
}
impl Prefix {
    fn capture(tokens: &[u32], media: &Identity) -> Result<Self> {
        let bytes = tokens.len().checked_mul(4).and_then(|n| n.checked_add(media.metadata_bytes()))
            .ok_or_else(|| anyhow::anyhow!("parent metadata extent"))?;
        let reservation = crate::memory::reserve_host_workspace(bytes.max(1))?;
        let mut copy = Vec::new();
        copy.try_reserve_exact(tokens.len())?;
        copy.extend_from_slice(tokens);
        Ok(Self { tokens: copy, media: media.clone(), _reservation: Some(reservation) })
    }
    fn covers(&self, tokens: &[u32], media: &Identity) -> bool {
        self.tokens.starts_with(tokens) && self.media.matches_prefix(media, tokens.len())
    }
}

#[derive(Clone, Default)]
enum Parent {
    #[default]
    None,
    Known(Rc<Prefix>),
    // If even the small CPU identity copy cannot retain the physical reserve,
    // keep all cache entries instead of silently abandoning parent protection.
    Unknown,
}
#[derive(Default)]
pub(super) struct Protection {
    recent: Parent,
    parent: Parent,
    epoch: bool,
}
impl Protection {
    pub(super) fn remember_foreground(&mut self, tokens: &[u32], media: &Identity) {
        self.recent = Prefix::capture(tokens, media).map_or(Parent::Unknown, |p| Parent::Known(Rc::new(p)));
    }
    pub(super) fn begin_epoch(&mut self) {
        if !self.epoch { self.parent = self.recent.clone(); self.epoch = true; }
    }
    pub(super) fn end_epoch(&mut self) { self.parent = Parent::None; self.epoch = false; }
    fn identity_mask(parent: &Parent, entries: &[(&[u32], &Identity)]) -> u16 {
        match parent {
            Parent::None => 0,
            Parent::Unknown => ((1u32 << entries.len()) - 1) as u16,
            Parent::Known(parent) => entries.iter().enumerate()
                .filter(|(_, (tokens, media))| parent.covers(tokens, media))
                .max_by_key(|(_, (tokens, _))| tokens.len())
                .map_or(0, |(i, _)| 1 << i),
        }
    }
    fn parent_mask(&self, entries: &[(&[u32], &Identity)]) -> u16 {
        // Keep the fan-out ancestor even when a foreground child completes.
        // Also keep the latest foreground turn: a long-lived background owner
        // may prevent this admission epoch from becoming completely idle.
        Self::identity_mask(&self.parent, entries) | Self::identity_mask(&self.recent, entries)
    }
    pub(super) fn mask(&self, w: &Worker, queue: &Queue,
        ready: &VecDeque<(Option<u64>, Job)>, images: &[ParkedRequest]) -> Result<u16> {
        ensure!(w.cached_sessions.len()<=w.session_cache_slots
            &&w.cached_sessions.len()<=MAX_SESSION_CACHE_SLOTS,"park protection cache bound");
        let entries = w.cached_sessions.iter().map(|c| (c.prefix.as_slice(), &c.media)).collect::<Vec<_>>();
        let mut protected = self.parent_mask(&entries);
        for (i, (tokens, media)) in entries.into_iter().enumerate() {
            if protected & (1 << i) != 0 { continue; }
            let mut peer_needs=false;
            for image in images {peer_needs|=image.needs_prefix(w,tokens,media)?;}
            if w.history.needs_prefix(tokens, media)
                || peer_needs
                || ready.iter().any(|(_, job)| job.needs_cached_prefix(tokens, media))
                || queue.pending_needs_prefix(tokens, media)? {
                protected |= 1 << i;
            }
        }
        Ok(protected)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn children_do_not_replace_the_frozen_parent_and_identity_includes_media() {
        // Policy tests run in a bounded CPU build container smaller than the
        // model's 8GiB reserve. Construct only identity metadata here; serving
        // always uses the guarded capture above and conservatively falls back
        // to Unknown if that reservation is refused.
        let remember = |protection: &mut Protection, tokens: &[u32], media: &Identity| {
            protection.recent = Parent::Known(Rc::new(Prefix {
                tokens: tokens.to_vec(), media: media.clone(), _reservation: None,
            }));
        };
        let empty = Identity::default();
        let mut protection = Protection::default();
        remember(&mut protection, &[10, 11, 12], &empty);
        protection.begin_epoch();
        let entries = [(&[10][..], &empty), (&[10,11][..], &empty), (&[20,21][..], &empty)];
        assert_eq!(protection.parent_mask(&entries), 2, "retain the longest usable parent");
        remember(&mut protection, &[20,21,22], &empty);
        assert_eq!(protection.parent_mask(&entries), 6, "keep both the burst parent and the newest foreground turn");
        protection.end_epoch(); protection.begin_epoch();
        assert_eq!(protection.parent_mask(&entries), 4);
        let tokens = [crate::prompt_media::IMAGE_TOKEN];
        let media = |byte| Identity::new(vec![crate::prompt_media::Placement {
            start:0, tokens:1, key:crate::prompt_media::ImageKey{input:[byte;32],encoder:[1;32]},
        }], &tokens).unwrap();
        let a=media(2); let b=media(3);
        protection.end_epoch(); remember(&mut protection, &tokens, &a); protection.begin_epoch();
        assert_eq!(protection.parent_mask(&[(&tokens,&b),(&tokens,&a)]), 2);
        protection.parent=Parent::Unknown;
        assert_eq!(protection.parent_mask(&entries), 7, "metadata refusal must preserve cache");
        let eight=(1..=8).map(|n|vec![10;n]).collect::<Vec<_>>();
        let entries=eight.iter().map(|t|(t.as_slice(),&empty)).collect::<Vec<_>>();
        assert_eq!(protection.parent_mask(&entries),255,"unknown parent protects all eight slots");
        protection.parent=Parent::None;
        remember(&mut protection,&[10;8],&empty);
        assert_eq!(protection.parent_mask(&entries),128,"longest parent can occupy the eighth bit");
        let sixteen=(1..=16).map(|n|vec![10;n]).collect::<Vec<_>>();
        let entries=sixteen.iter().map(|t|(t.as_slice(),&empty)).collect::<Vec<_>>();
        protection.parent=Parent::Unknown;
        assert_eq!(protection.parent_mask(&entries),u16::MAX,"unknown parent protects all sixteen slots");
        protection.parent=Parent::None;
        remember(&mut protection,&[10;16],&empty);
        assert_eq!(protection.parent_mask(&entries),1<<15,"longest parent can occupy the high bit");
    }
}
