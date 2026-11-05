//! Integer history/visibility contract for a resident single-stream worker.
//! Model/profile leases remain authoritative for GPU state. This book admits
//! exact full-prefix reuse only; it cannot roll back or reconstruct GPU state.
use anyhow::{Result, ensure};
use crate::prompt_media::Identity as MediaIdentity;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Finish {
    Stop,
    Length,
    Cancelled,
}

#[derive(Clone, Debug)]
pub(crate) struct Plan {
    pub id: u64,
    pub seed: u64,
    pub limit: usize,
    pub previous: usize,
    pub reuse: bool,
    /// The client replaces/regenerates a naturally completed output while
    /// retaining its exact input. Only an optional displaced-output copy may
    /// be skipped; this does not authorize any different GPU prefix reuse.
    pub replaces_completed_output: bool,
    pub prompt: Vec<u32>,
    pub media: MediaIdentity,
}
impl Plan {
    pub fn delta(&self) -> &[u32] {
        &self.prompt[if self.reuse { self.previous } else { 0 }..]
    }
}

#[derive(Clone, Debug)]
enum Phase {
    Idle,
    Prefilling {
        id: u64,
        prompt: Vec<u32>,
        media: MediaIdentity,
        limit: usize,
    },
    Generating {
        id: u64,
        limit: usize,
        emitted: usize,
    },
    Poisoned,
}

#[derive(Clone)]
pub(crate) struct ResidentHistory {
    capacity: usize,
    eos: u32,
    last_id: u64,
    committed: Vec<u32>,
    media: MediaIdentity,
    pending: Option<u32>,
    phase: Phase,
    completed_prompt_tokens: Option<usize>,
}

impl ResidentHistory {
    /// One instance belongs to one fixed model/tokenizer/numerical-profile
    /// identity and writer group. Changing that identity requires a new owner.
    pub fn new(capacity: usize, eos: u32) -> Result<Self> {
        ensure!(
            (1..=crate::config::MODEL_CONTEXT).contains(&capacity) && eos < 129280,
            "resident history geometry"
        );
        Ok(Self {
            capacity,
            eos,
            last_id: 0,
            committed: Vec::new(),
            media: MediaIdentity::default(),
            pending: None,
            phase: Phase::Idle,
            completed_prompt_tokens: None,
        })
    }
    pub fn prepare(&mut self, id: u64, prompt: Vec<u32>, limit: usize, seed: u64) -> Result<Plan> {
        self.prepare_with_media(id, prompt, MediaIdentity::default(), limit, seed)
    }
    pub(crate) fn prepare_with_media(&mut self, id: u64, prompt: Vec<u32>, media: MediaIdentity, limit: usize, seed: u64) -> Result<Plan> {
        ensure!(
            matches!(self.phase, Phase::Idle),
            "resident owner is active or poisoned"
        );
        ensure!(
            id > self.last_id
                && seed < (1u64 << 63)
                && !prompt.is_empty()
                && prompt.iter().all(|&id| id < 129280
                    && (!media.is_empty() || id != crate::prompt_media::IMAGE_TOKEN))
                && (1..=262144).contains(&limit)
                && prompt
                    .len()
                    .checked_add(limit)
                    .is_some_and(|n| n <= self.capacity),
            "request identity, sampling limit, token domain, or context admission"
        );
        if !media.is_empty() { media.validate_prompt(&prompt)?; }
        let previous = self.committed.len();
        let reuse = previous > 0 && prompt.len() > previous && prompt.starts_with(&self.committed)
            && media.matches_prefix(&self.media, previous);
        let replaces_completed_output = !reuse && self.completed_prompt_tokens.is_some_and(|n|
            n < previous && prompt.starts_with(&self.committed[..n])
            && media.matches_prefix(&self.media,n));
        // The final emitted anchor was never in KV. The incoming prompt may
        // include it or replace it; only committed IDs govern reusable state.
        let plan = Plan {
            id,
            seed,
            limit,
            previous,
            reuse,
            replaces_completed_output,
            prompt,
            media,
        };
        self.phase = Phase::Prefilling {
            id,
            prompt: plan.prompt.clone(),
            media: plan.media.clone(),
            limit,
        };
        self.last_id = id;
        self.completed_prompt_tokens = None;
        Ok(plan)
    }
    /// Record a committed partial prompt without emitting its temporary anchor.
    /// The complete original prompt remains owned until final prefill or cancel.
    pub(crate) fn partial_prefilled(&mut self,frontier:usize,anchor:u32)->Result<()> {
        let Phase::Prefilling{prompt,media,..}=&self.phase else {
            anyhow::bail!("partial progress requires a prefilling request")
        };
        ensure!(frontier>0 && frontier<prompt.len() && anchor<129280,
            "partial prompt frontier or anchor");
        self.committed=prompt[..frontier].to_vec();
        self.media=media.through(frontier);
        self.pending=Some(anchor);
        Ok(())
    }
    pub(crate) fn owned_token_bytes(&self)->usize {
        self.committed.len()*4+self.media.metadata_bytes()+match &self.phase {
            Phase::Prefilling{prompt,media,..}=>prompt.len()*4+media.metadata_bytes(),_=>0}
    }
    pub fn prefilled(&mut self, frontier: usize, anchor: u32) -> Result<Vec<u32>> {
        let phase = std::mem::replace(&mut self.phase, Phase::Poisoned);
        let Phase::Prefilling { id, prompt, media, limit } = phase else {
            anyhow::bail!("unexpected resident prefill completion")
        };
        ensure!(
            frontier == prompt.len() && anchor < 129280,
            "resident prefill frontier/anchor mismatch"
        );
        self.committed = prompt;
        self.media = media;
        self.pending = Some(anchor);
        self.phase = Phase::Generating {
            id,
            limit,
            emitted: 1,
        };
        Ok(vec![anchor])
    }
    /// Adopt a same-owner cached accepted prefix while the already validated
    /// new request is prefilling. Request IDs remain monotonic; only causal
    /// token history changes, after the worker restores its matching GPU state.
    pub(crate) fn restore_cached_prefix(&mut self,plan:&mut Plan,committed:&[u32],pending:u32)->Result<()> {
        self.restore_cached_media(plan,committed,Some(pending),false,&MediaIdentity::default())
    }
    /// Complete prompt restoration is allowed only when the worker also owns
    /// that prompt's saved logits and will resample its first uncommitted token.
    pub(crate) fn restore_cached_prompt(&mut self,plan:&mut Plan,committed:&[u32],pending:u32)->Result<()> {
        self.restore_cached_media(plan,committed,Some(pending),true,&MediaIdentity::default())
    }
    /// Adopt an encoder cache hit only after the worker completes its decoder
    /// suffix. No invented pending token is exposed for an incomplete image.
    pub(crate) fn restore_encoder_prefix(&mut self,plan:&mut Plan,committed:&[u32])->Result<()> {
        self.restore_cached_media(plan,committed,None,false,&MediaIdentity::default())
    }
    pub(crate) fn restore_cached_media(&mut self,plan:&mut Plan,committed:&[u32],pending:Option<u32>,exact:bool,cached_media:&MediaIdentity)->Result<()> {
        let Phase::Prefilling{id,prompt,media,limit}=&self.phase else {
            anyhow::bail!("cached prefix requires a pending prefill")
        };
        ensure!(*id==plan.id && *limit==plan.limit && *prompt==plan.prompt && *media==plan.media && !plan.reuse
            && !committed.is_empty() && (if exact {committed.len()==prompt.len()}else{committed.len()<prompt.len()})
            && prompt.starts_with(committed)
            && media.matches_prefix(cached_media,committed.len())
            && pending.is_none_or(|p|p<129280) && committed.iter().all(|&v|v<129280),"cached resident prefix admission");
        self.committed=committed.to_vec();self.pending=pending;
        self.media=cached_media.through(committed.len());
        plan.previous=committed.len();plan.reuse=true;
        Ok(())
    }
    pub fn step_limit(&self) -> Result<usize> {
        self.step_limit_for_rows(6)
    }
    pub(crate) fn step_limit_for_rows(&self, rows: usize) -> Result<usize> {
        ensure!([6,16,32].contains(&rows), "resident verification capacity");
        let Phase::Generating { limit, emitted, .. } = self.phase else {
            anyhow::bail!("resident request is not generating")
        };
        Ok(if self.pending == Some(self.eos) {
            0
        } else {
            (limit - emitted).min(rows)
        })
    }
    /// Called only after target AND drafter commit and rank agreement. The
    /// accepted input prefix includes the old pending anchor; emission omits it
    /// and includes the newly sampled, still-uncommitted anchor exactly once.
    pub fn committed_step(
        &mut self,
        inputs: &[u32],
        next: u32,
        frontier: usize,
    ) -> Result<Vec<u32>> {
        let phase = std::mem::replace(&mut self.phase, Phase::Poisoned);
        let Phase::Generating { id, limit, emitted } = phase else {
            anyhow::bail!("unexpected resident commit")
        };
        ensure!(
            !inputs.is_empty()
                && inputs.len() <= 32
                && inputs.len() <= limit - emitted
                && inputs.first() == self.pending.as_ref()
                && inputs.iter().all(|&t| t < 129280 && t != self.eos)
                && next < 129280
                && frontier <= self.capacity
                && self.committed.len().checked_add(inputs.len()) == Some(frontier),
            "resident committed-prefix or emission frontier mismatch"
        );
        self.committed.extend_from_slice(inputs);
        self.pending = Some(next);
        self.phase = Phase::Generating {
            id,
            limit,
            emitted: emitted + inputs.len(),
        };
        let mut output = inputs[1..].to_vec();
        output.push(next);
        Ok(output)
    }
    pub fn finish(&mut self, reason: Finish) -> Result<()> {
        let phase = std::mem::replace(&mut self.phase, Phase::Poisoned);
        if matches!(phase,Phase::Prefilling{..}) && reason==Finish::Cancelled {
            self.phase=Phase::Idle;
            return Ok(());
        }
        let Phase::Generating { limit, emitted, .. } = phase else {
            anyhow::bail!("unexpected resident finish")
        };
        ensure!(
            match reason {
                Finish::Stop => self.pending == Some(self.eos),
                Finish::Length => emitted == limit && self.pending != Some(self.eos),
                Finish::Cancelled => true,
            },
            "backend completion reason contradicts committed state"
        );
        self.completed_prompt_tokens = if reason == Finish::Stop {
            Some(self.committed.len().checked_sub(emitted - 1)
                .ok_or_else(||anyhow::anyhow!("completed request input extent"))?)
        } else { None };
        self.phase = Phase::Idle;
        Ok(())
    }
    pub fn retire(&mut self) {
        self.phase = Phase::Poisoned;
    }
    /// Physical owners keep their phase and token buffers in place. Switching
    /// the selected owner must still preserve the group-wide request-ID floor.
    pub(crate) fn synchronize_request_ids(&mut self,other:&mut Self)->Result<()> {
        ensure!(self.capacity==other.capacity&&self.eos==other.eos
            &&!matches!(self.phase,Phase::Poisoned)&&!matches!(other.phase,Phase::Poisoned),
            "resident bank identity/phase");
        let newest=self.last_id.max(other.last_id);
        self.last_id=newest;other.last_id=newest;Ok(())
    }
    /// Keep the currently loaded model's token identity while another request
    /// is selected. The caller must already own a complete GPU/sampler image.
    pub(crate) fn suspend(&mut self) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Generating { .. } | Phase::Prefilling { .. }) && self.pending.is_some(),
            "suspension requires an active request");
        self.phase = Phase::Idle;
        Ok(())
    }
    /// Restore request-local visibility only after the matching GPU image was
    /// restored. Global request IDs must never rewind with an older request.
    pub(crate) fn resume(&mut self, saved: &Self) -> Result<()> {
        ensure!(matches!(self.phase, Phase::Idle)
            && matches!(saved.phase, Phase::Generating { .. } | Phase::Prefilling { .. }) && saved.pending.is_some()
            && self.capacity == saved.capacity && self.eos == saved.eos,
            "resumed request history ownership/phase");
        let newest_id = self.last_id.max(saved.last_id);
        *self = saved.clone();
        self.last_id = newest_id;
        Ok(())
    }
    /// Both requests stay admitted. Move host ownership with the matching GPU
    /// exchange, preserving the global request-ID high-water mark on both.
    pub(crate) fn exchange(&mut self,saved:&mut Self)->Result<()> {
        let active=|h:&Self|matches!(h.phase,Phase::Generating{..}|Phase::Prefilling{..})&&h.pending.is_some();
        ensure!(active(self)&&active(saved)&&self.capacity==saved.capacity&&self.eos==saved.eos,
            "exchanged request history ownership/phase");
        let newest=self.last_id.max(saved.last_id);std::mem::swap(self,saved);
        self.last_id=newest;saved.last_id=newest;Ok(())
    }
    /// A partially prefilling request still needs its complete original input,
    /// including any cached prefix beyond its current committed frontier.
    pub(crate) fn needs_prefix(&self, prefix: &[u32], media: &MediaIdentity) -> bool {
        let (tokens, identity) = match &self.phase {
            Phase::Prefilling { prompt, media, .. } => (prompt.as_slice(), media),
            _ => (self.committed.as_slice(), &self.media),
        };
        tokens.starts_with(prefix) && identity.matches_prefix(media, prefix.len())
    }
    pub fn frontier(&self) -> usize {
        self.committed.len()
    }
    pub fn committed(&self) -> &[u32] {
        &self.committed
    }
    pub(crate) fn media(&self) -> &MediaIdentity { &self.media }
    pub fn pending(&self) -> Option<u32> {
        self.pending
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn only_replaced_naturally_finished_output_is_an_optional_copy_hint() {
        let finished=|reason| {
            let mut h=ResidentHistory::new(128,1).unwrap();
            h.prepare(1,vec![2,3],3,7).unwrap();h.prefilled(2,10).unwrap();
            h.committed_step(&[10,11],if reason==Finish::Stop{1}else{12},4).unwrap();
            h.finish(reason).unwrap();h
        };
        for prompt in [vec![2,3],vec![2,3,10,55,7]] {
            let p=finished(Finish::Stop).prepare(2,prompt,3,8).unwrap();
            assert!(!p.reuse&&p.replaces_completed_output);
        }
        let p=finished(Finish::Stop).prepare(2,vec![2,3,10,11,1,7],3,8).unwrap();
        assert!(p.reuse&&!p.replaces_completed_output);
        for reason in [Finish::Length,Finish::Cancelled] {
            assert!(!finished(reason).prepare(2,vec![2,3,10,55,7],3,8).unwrap().replaces_completed_output);
        }
        assert!(!finished(Finish::Stop).prepare(2,vec![2,4,10,55],3,8).unwrap().replaces_completed_output);
    }
    #[test]
    fn completed_output_hint_respects_media_identity_and_request_ownership() {
        use crate::prompt_media::IMAGE_TOKEN as I;
        let prompt=vec![2,I,I,I,3];let identity=media(&prompt,1);
        let mut h=ResidentHistory::new(128,1).unwrap();
        h.prepare_with_media(1,prompt.clone(),identity.clone(),3,7).unwrap();
        h.prefilled(5,10).unwrap();h.committed_step(&[10,11],1,7).unwrap();
        let saved=h.clone();h.suspend().unwrap();
        assert!(!h.prepare_with_media(2,prompt.clone(),identity.clone(),3,8).unwrap().replaces_completed_output);
        h.prefilled(5,15).unwrap();h.finish(Finish::Cancelled).unwrap();
        h.resume(&saved).unwrap();h.finish(Finish::Stop).unwrap();
        let mut different=h.clone();
        assert!(!different.prepare_with_media(3,prompt.clone(),media(&prompt,2),3,9).unwrap().replaces_completed_output);
        assert!(h.prepare_with_media(3,prompt.clone(),identity,3,9).unwrap().replaces_completed_output);
        h.partial_prefilled(1,15).unwrap();h.finish(Finish::Cancelled).unwrap();
        assert!(!h.prepare(4,vec![2,5],1,10).unwrap().replaces_completed_output);
    }
    fn media(prompt: &[u32], key: u8) -> MediaIdentity {
        use crate::prompt_media::{ImageKey, Placement};
        MediaIdentity::new(vec![Placement { start:1, tokens:3,
            key:ImageKey {input:[key;32],encoder:[7;32]} }],prompt).unwrap()
    }
    #[test]
    fn identical_placeholder_tokens_do_not_reuse_different_images() {
        use crate::prompt_media::IMAGE_TOKEN as I;
        let first=vec![2,I,I,I,3]; let next=vec![2,I,I,I,3,4];
        let a=media(&first,1);let b=media(&first,2);
        let mut h=ResidentHistory::new(128,1).unwrap();
        assert!(h.prepare(1,first.clone(),1,7).is_err());
        h.prepare_with_media(1,first.clone(),a.clone(),1,7).unwrap();
        h.prefilled(5,8).unwrap();h.finish(Finish::Length).unwrap();
        let plan=h.prepare_with_media(2,next.clone(),a.clone(),1,8).unwrap();
        assert!(plan.reuse);h.partial_prefilled(5,9).unwrap();h.finish(Finish::Cancelled).unwrap();
        let mut plan=h.prepare_with_media(3,next,b.clone(),1,9).unwrap();
        assert!(!plan.reuse);
        assert!(h.restore_cached_prefix(&mut plan,&first,8).is_err());
        assert!(h.restore_cached_media(&mut plan,&first,Some(8),false,&a).is_err());
        h.restore_cached_media(&mut plan,&first,Some(8),false,&b).unwrap();
        assert!(plan.reuse);assert_eq!(h.media(),&b);
        h.prefilled(6,10).unwrap();h.finish(Finish::Length).unwrap();
    }
    #[test]
    fn partial_image_identity_survives_suspend_and_cannot_be_restored_as_text() {
        use crate::prompt_media::IMAGE_TOKEN as I;
        let prompt=vec![2,I,I,I,3];let a=media(&prompt,1);
        let mut h=ResidentHistory::new(128,1).unwrap();
        h.prepare_with_media(1,prompt.clone(),a.clone(),2,7).unwrap();
        h.partial_prefilled(2,9).unwrap();
        assert_eq!(h.media().placements()[0].tokens,3);
        assert!(h.owned_token_bytes()>(prompt.len()+2)*4);
        assert!(h.needs_prefix(&prompt[..4],&a), "a pending prompt needs prefixes past its committed frontier");
        assert!(!h.needs_prefix(&prompt[..4],&media(&prompt,2)), "equal image placeholders are not equal input");
        let saved=h.clone();h.suspend().unwrap();
        assert!(saved.needs_prefix(&prompt[..4],&a), "the parked original input retains its needed prefixes");
        h.prepare(2,vec![20,21],1,8).unwrap();h.prefilled(2,22).unwrap();h.finish(Finish::Length).unwrap();
        h.resume(&saved).unwrap();assert_eq!(h.media(),&a);h.finish(Finish::Cancelled).unwrap();
        let mut plan=h.prepare_with_media(3,prompt.clone(),media(&prompt,2),1,9).unwrap();
        assert!(!plan.reuse);
        assert!(h.restore_encoder_prefix(&mut plan,&prompt[..2]).is_err());
        // The text-only prefix before the image is reusable for either image.
        h.restore_encoder_prefix(&mut plan,&prompt[..1]).unwrap();
        assert!(h.media().is_empty());h.prefilled(5,11).unwrap();
    }
    #[test]
    fn complete_prompt_restore_requires_explicit_admission_and_a_fresh_anchor() {
        let mut h=ResidentHistory::new(32768,129000).unwrap();
        h.prepare(1,vec![1,2],1,8).unwrap();
        h.prefilled(2,7).unwrap();h.finish(Finish::Length).unwrap();
        let mut plan=h.prepare(2,vec![1,2],2,9).unwrap();
        assert!(!plan.reuse);
        assert!(h.restore_cached_prefix(&mut plan,&[1,2],7).is_err());
        h.restore_cached_prompt(&mut plan,&[1,2],7).unwrap();
        assert!(plan.reuse && plan.delta().is_empty() && plan.seed==9);
        assert_eq!(h.prefilled(2,11).unwrap(),vec![11]);
        assert_eq!(h.pending(),Some(11));
    }
    use super::*;
    #[test]
    fn partial_prompt_survives_foreground_without_emitting_its_temporary_anchor() {
        let mut h=ResidentHistory::new(128,1).unwrap();
        h.prepare(1,vec![2,3,4,5,6,7,8,9],3,7).unwrap();
        h.partial_prefilled(4,1).unwrap(); // Even an EOS metadata anchor is not output.
        assert!(h.needs_prefix(&[2,3,4,5,6,7],&MediaIdentity::default()));
        assert!(!h.needs_prefix(&[2,3,4,5,6,8],&MediaIdentity::default()));
        assert_eq!(h.owned_token_bytes(),(8+4)*4);
        assert!(h.step_limit().is_err());
        let saved=h.clone();h.suspend().unwrap();
        h.prepare(2,vec![20,21],1,8).unwrap();
        assert_eq!(h.prefilled(2,22).unwrap(),vec![22]);h.finish(Finish::Length).unwrap();
        h.resume(&saved).unwrap();
        h.partial_prefilled(6,42).unwrap();
        assert!(h.step_limit().is_err());
        assert_eq!(h.prefilled(8,10).unwrap(),vec![10]);
        assert_eq!(h.step_limit().unwrap(),2);
        assert_eq!(h.committed(),[2,3,4,5,6,7,8,9]);
        h.finish(Finish::Cancelled).unwrap();
        assert!(h.prepare(2,vec![11,12],2,9).is_err());
        h.prepare(3,vec![11,12],2,9).unwrap();h.partial_prefilled(1,13).unwrap();
        h.finish(Finish::Cancelled).unwrap();
        assert!(h.prepare(4,vec![14,15],2,10).is_ok());
    }
    #[test]
    fn suspended_request_keeps_budget_anchor_and_global_request_identity() {
        let mut h=ResidentHistory::new(128,1).unwrap();
        h.prepare(1,vec![2,3],5,7).unwrap();h.prefilled(2,4).unwrap();
        assert_eq!(h.committed_step(&[4],5,3).unwrap(),vec![5]);
        let saved=h.clone();h.suspend().unwrap();
        h.prepare(2,vec![20,21],1,8).unwrap();h.prefilled(2,22).unwrap();
        assert!(h.resume(&saved).is_err());
        h.finish(Finish::Length).unwrap();h.resume(&saved).unwrap();
        assert_eq!(h.frontier(),3);assert_eq!(h.pending(),Some(5));
        assert_eq!(h.step_limit().unwrap(),3);
        assert_eq!(h.committed_step(&[5,6,7],8,6).unwrap(),vec![6,7,8]);
        assert_eq!(h.step_limit().unwrap(),0);h.finish(Finish::Length).unwrap();
        assert!(h.prepare(2,vec![30],1,9).is_err());
        h.prepare(3,vec![30],1,9).unwrap();
    }
    #[test]
    fn exchange_moves_budgets_and_preserves_global_id_after_either_owner_finishes(){
        let mut a=ResidentHistory::new(128,1).unwrap();
        a.prepare(1,vec![2,3],5,7).unwrap();a.prefilled(2,4).unwrap();
        a.committed_step(&[4],5,3).unwrap();
        let mut b=a.clone();b.suspend().unwrap();
        b.prepare(2,vec![20,21,22],3,8).unwrap();b.prefilled(3,23).unwrap();
        a.exchange(&mut b).unwrap();
        assert_eq!(a.committed(),[20,21,22]);assert_eq!(a.pending(),Some(23));assert_eq!(a.step_limit().unwrap(),2);
        assert_eq!(b.committed(),[2,3,4]);assert_eq!(b.pending(),Some(5));assert_eq!(b.step_limit().unwrap(),3);
        a.exchange(&mut b).unwrap();a.finish(Finish::Cancelled).unwrap();b.finish(Finish::Cancelled).unwrap();
        assert!(a.exchange(&mut b).is_err());
        assert!(a.prepare(2,vec![30],1,9).is_err());assert!(b.prepare(2,vec![30],1,9).is_err());
        a.prepare(3,vec![30],1,9).unwrap();b.prepare(3,vec![31],1,10).unwrap();
    }
    #[test]
    fn physical_banks_preserve_progress_and_global_ids_without_snapshot_copies(){
        let mut a=ResidentHistory::new(128,1).unwrap();
        let mut b=ResidentHistory::new(128,1).unwrap();
        a.prepare(1,vec![2,3],5,7).unwrap();a.prefilled(2,4).unwrap();
        a.committed_step(&[4],5,3).unwrap();
        let storage=a.committed.as_ptr();
        a.synchronize_request_ids(&mut b).unwrap();
        assert!(b.prepare(1,vec![20,21],3,8).is_err());
        b.prepare(2,vec![20,21],3,8).unwrap();b.partial_prefilled(1,22).unwrap();
        b.synchronize_request_ids(&mut a).unwrap();
        assert_eq!(a.committed.as_ptr(),storage);
        assert_eq!(a.committed(),[2,3,4]);assert_eq!(a.pending(),Some(5));
        assert_eq!(a.step_limit().unwrap(),3);
        b.finish(Finish::Cancelled).unwrap();
        a.finish(Finish::Cancelled).unwrap();
        assert!(a.prepare(2,vec![30],1,9).is_err());
        a.prepare(3,vec![30],1,9).unwrap();a.prefilled(1,31).unwrap();
        a.synchronize_request_ids(&mut b).unwrap();
        assert!(b.prepare(3,vec![40],1,10).is_err());
        b.prepare(4,vec![40],1,10).unwrap();
        a.retire();assert!(a.synchronize_request_ids(&mut b).is_err());
    }
    #[test]
    fn encoder_prefix_adoption_has_no_invented_anchor() {
        let mut h=ResidentHistory::new(1024,1).unwrap();
        h.prepare(1,vec![20,21],1,7).unwrap();h.prefilled(2,22).unwrap();h.finish(Finish::Length).unwrap();
        let mut plan=h.prepare(2,vec![2,3,4,5],2,8).unwrap();
        assert!(h.restore_encoder_prefix(&mut plan,&[2,9]).is_err());
        assert_eq!(h.pending(),Some(22));
        h.restore_encoder_prefix(&mut plan,&[2,3]).unwrap();
        assert_eq!(h.pending(),None);assert_eq!(plan.delta(),[4,5]);
        assert!(h.step_limit().is_err());
        assert_eq!(h.prefilled(4,6).unwrap(),vec![6]);
        assert_eq!(h.pending(),Some(6));assert_eq!(h.step_limit().unwrap(),1);
    }

    #[test]
    fn cached_parent_prefix_restores_without_rewinding_request_identity() {
        let mut h=ResidentHistory::new(128,1).unwrap();
        h.prepare(1,vec![2,3,4],2,7).unwrap();h.prefilled(3,5).unwrap();
        h.committed_step(&[5],6,4).unwrap();h.finish(Finish::Length).unwrap();
        let parent=h.committed().to_vec();let anchor=h.pending().unwrap();
        h.prepare(2,vec![20,21],1,8).unwrap();h.prefilled(2,22).unwrap();h.finish(Finish::Length).unwrap();
        let mut plan=h.prepare(3,vec![2,3,4,5,6,7],1,9).unwrap();assert!(!plan.reuse);
        assert!(h.restore_cached_prefix(&mut plan,&[9,9],anchor).is_err());
        assert_eq!(h.frontier(),2);assert!(!plan.reuse);
        h.restore_cached_prefix(&mut plan,&parent,anchor).unwrap();
        assert_eq!(plan.delta(),[6,7]);assert_eq!(h.pending(),Some(6));
        h.prefilled(6,8).unwrap();h.finish(Finish::Length).unwrap();
        assert!(h.prepare(2,vec![3],1,9).is_err());
        assert!(h.restore_cached_prefix(&mut plan,&parent,anchor).is_err());
    }
    #[test]
    fn input_refusal_before_prefill_preserves_the_current_or_restored_owner() {
        let mut h=ResidentHistory::new(128,1).unwrap();
        // The first request may be refused before any GPU prompt exists.
        h.prepare(1,vec![2,3],8,7).unwrap();
        h.finish(Finish::Cancelled).unwrap();
        assert_eq!(h.frontier(),0);assert_eq!(h.pending(),None);
        assert!(h.prepare(1,vec![2,3],8,7).is_err());
        h.prepare(2,vec![2,3],1,7).unwrap();
        h.prefilled(2,4).unwrap();h.finish(Finish::Length).unwrap();
        h.prepare(3,vec![8,9],8,7).unwrap();
        h.finish(Finish::Cancelled).unwrap();
        assert_eq!(h.committed(),[2,3]);assert_eq!(h.pending(),Some(4));
        // An already completed cache restoration is a valid retained owner
        // even if the new image's workspace subsequently fails admission.
        let mut plan=h.prepare(4,vec![8,9,10,11],8,7).unwrap();
        h.restore_cached_prefix(&mut plan,&[8,9],10).unwrap();
        h.finish(Finish::Cancelled).unwrap();
        assert_eq!(h.committed(),[8,9]);assert_eq!(h.pending(),Some(10));
        let next=h.prepare(5,vec![8,9,10,12],1,7).unwrap();
        assert!(next.reuse);assert_eq!(next.delta(),[10,12]);
        h.prefilled(4,13).unwrap();h.finish(Finish::Length).unwrap();
    }
    #[test]
    fn exact_retention_includes_pending_anchor_once_and_branches_reset() {
        let mut h = ResidentHistory::new(128, 1).unwrap();
        let p = h.prepare(1, vec![0, 42], 8, 7).unwrap();
        assert!(!p.reuse);
        assert_eq!(p.delta(), [0, 42]);
        assert_eq!(h.prefilled(2, 50).unwrap(), [50]);
        assert_eq!(
            h.committed_step(&[50, 51, 52], 53, 5).unwrap(),
            [51, 52, 53]
        );
        h.finish(Finish::Cancelled).unwrap();
        let p = h.prepare(2, vec![0, 42, 50, 51, 52, 53, 60], 4, 8).unwrap();
        assert!(p.reuse);
        assert_eq!(p.delta(), [53, 60]);
        h.prefilled(7, 70).unwrap();
        h.committed_step(&[70], 1, 8).unwrap();
        h.finish(Finish::Stop).unwrap();
        let p = h
            .prepare(3, vec![0, 42, 50, 51, 52, 53, 60, 70, 99], 1, 9)
            .unwrap();
        assert!(p.reuse);
        assert_eq!(
            p.delta(),
            [99],
            "a new prompt may replace the uncommitted EOS"
        );
        h.prefilled(9, 88).unwrap();
        assert_eq!(h.step_limit().unwrap(), 0);
        h.finish(Finish::Length).unwrap();
        let p = h.prepare(4, vec![0, 42, 50, 51], 2, 10).unwrap();
        assert!(
            !p.reuse,
            "a shorter prefix needs a real reset, not an invented rollback"
        );
        assert_eq!(p.delta(), p.prompt);
    }
    #[test]
    fn invalid_execution_frontiers_permanently_retire_the_owner() {
        let mut h = ResidentHistory::new(128, 1).unwrap();
        assert!(h.prepare(1, vec![0, 42], 127, 7).is_err());
        h.prepare(1, vec![0, 42], 8, 7).unwrap();
        h.prefilled(2, 50).unwrap();
        assert!(h.committed_step(&[50, 1, 51], 52, 5).is_err());
        assert_eq!(h.frontier(), 2);
        assert_eq!(h.pending(), Some(50));
        assert!(h.prepare(2, vec![0, 42], 8, 7).is_err());
        assert!(h.finish(Finish::Cancelled).is_err());
    }
    #[test]
    fn all_eos_positions_and_budget_edges_preserve_one_pending_token() {
        for n in 1..=32 {
            let mut h = ResidentHistory::new(128, 1).unwrap();
            h.prepare(1, vec![0, 42], n + 1, 7).unwrap();
            h.prefilled(2, 50).unwrap();
            assert_eq!(h.step_limit_for_rows(16).unwrap(), n.min(16));
            assert_eq!(h.step_limit_for_rows(32).unwrap(), n);
            assert_eq!(h.step_limit().unwrap(), n.min(6));
            let input: Vec<_> = (50..50 + n as u32).collect();
            let emitted = h.committed_step(&input, 1, 2 + n).unwrap();
            assert_eq!(emitted.len(), n);
            assert_eq!(emitted.last(), Some(&1));
            assert_eq!(h.frontier(), 2 + n);
            assert_eq!(h.step_limit().unwrap(), 0);
            h.finish(Finish::Stop).unwrap();
        }
        let mut h = ResidentHistory::new(128, 1).unwrap();
        h.prepare(1, vec![0, 42], 1, 7).unwrap();
        h.prefilled(2, 1).unwrap();
        assert_eq!(h.frontier(), 2);
        h.finish(Finish::Stop).unwrap();
        assert!(
            h.prepare(1, vec![0, 42], 8, 7).is_err(),
            "request IDs cannot be replayed"
        );
    }
    #[test]
    fn oversized_blocks_and_remaining_budget_overruns_publish_nothing() {
        for (rows,limit) in [(33usize,64usize),(32,32)] {
            let mut h=ResidentHistory::new(128,1).unwrap();
            h.prepare(1,vec![0,42],limit,7).unwrap();h.prefilled(2,50).unwrap();
            let inputs=(50..50+rows as u32).collect::<Vec<_>>();
            assert!(h.committed_step(&inputs,90,2+rows).is_err());
            assert_eq!(h.frontier(),2);assert_eq!(h.pending(),Some(50));
            assert!(h.finish(Finish::Length).is_err());
        }
    }
}
