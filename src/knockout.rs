//! Plan-screen knockouts (CANADA_SCRATCH_KNOCKOUT_SCREEN). One arm of the L2 queue plan screen skips the launches of
//! one kernel group of the decode layer. The screen times the same verification step over and over, so a skipped
//! kernel's outputs still hold the values of the previous run of that step and everything downstream computes on
//! identical data (same routes, same bytes read elsewhere). The arm's gain is what the step would gain if that
//! group, with its weight reads, cost nothing: an upper bound for any work on it, measured in the model.
//! Timing experiments only. Nothing here is reachable outside the screen: the mask is zero unless the probe sets it.
use std::cell::Cell;
thread_local! {
    static MASK: Cell<u32> = const { Cell::new(0) };
}
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u32)]
pub(crate) enum Site {
    /// The FFN boundary's mixing-statistics pair (partial sums and reduction / Sinkhorn).
    StatsFfn = 1,
    /// The attention boundary's pair.
    StatsAttn = 2,
    /// Both boundaries' post-mix kernels.
    Post = 4,
    /// The QKV-A input projection.
    QkvA = 8,
    /// The QKV gather (collective).
    Gather = 16,
    /// The query projection wq_b.
    WqB = 32,
    /// The attention kernels (decode, merge) and the output's position rotation.
    Attention = 64,
    /// The output projection's first factor (wo_a partial product and its reduction).
    WoA = 128,
    /// Its second factor (wo_b).
    WoB = 256,
    /// The router's gate product and top-k.
    Router = 512,
    /// The attention and FFN sums (collectives).
    Sums = 1024,
}
pub(crate) fn has(site: Site) -> bool { MASK.with(|m| m.get() & site as u32 != 0) }
pub(crate) fn set(mask: u32) { MASK.with(|m| m.set(mask)); }
/// A knockout by name; `+` joins several ("statsffn+qkva").
pub(crate) fn parse(name: &str) -> anyhow::Result<u32> {
    let mut mask = 0u32;
    for part in name.split('+') {
        mask |= match part {
            "statsffn" => Site::StatsFfn as u32,
            "statsattn" => Site::StatsAttn as u32,
            "stats" => Site::StatsFfn as u32 | Site::StatsAttn as u32,
            "post" => Site::Post as u32,
            "qkva" => Site::QkvA as u32,
            "gather" => Site::Gather as u32,
            "wqb" => Site::WqB as u32,
            "attn" => Site::Attention as u32,
            "woa" => Site::WoA as u32,
            "wob" => Site::WoB as u32,
            "router" => Site::Router as u32,
            "sums" => Site::Sums as u32,
            other => anyhow::bail!("unknown knockout {other:?}"),
        };
    }
    anyhow::ensure!(mask != 0, "empty knockout");
    Ok(mask)
}
/// CANADA_SCRATCH_KNOCKOUT_SCREEN: comma-separated knockouts for the plan screen (at most sixteen arms).
pub(crate) fn screen() -> anyhow::Result<Vec<(String, u32)>> {
    match std::env::var("CANADA_SCRATCH_KNOCKOUT_SCREEN") {
        Ok(value) => {
            let arms = value.split(',').map(|name| Ok((name.to_string(), parse(name)?))).collect::<anyhow::Result<Vec<_>>>()?;
            anyhow::ensure!(!arms.is_empty() && arms.len() <= 16, "knockout screen takes one to sixteen arms");
            Ok(arms)
        }
        Err(std::env::VarError::NotPresent) => Ok(Vec::new()),
        Err(error) => Err(error.into()),
    }
}
#[cfg(test)]
mod tests {
    #[test]
    fn knockouts_parse_strictly() {
        assert_eq!(super::parse("statsffn").unwrap(), 1);
        assert_eq!(super::parse("stats").unwrap(), 3);
        assert_eq!(super::parse("statsffn+qkva").unwrap(), 9);
        assert_eq!(super::parse("sums+gather").unwrap(), 1040);
        for bad in ["", "x", "stats+", "qkva,woa"] { assert!(super::parse(bad).is_err(), "{bad}"); }
    }
}
