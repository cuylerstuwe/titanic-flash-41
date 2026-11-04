//! Explicit candidate controls for lossless distributed sampler transport.
//! These select implementations, never target precision or sampling settings.
use anyhow::{Result, ensure};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Policy {
    pub hierarchical_draft: bool,
    pub local_target: bool,
    pub local_history: bool,
    pub bundled_ids: bool,
    pub fused_target_controls: bool,
    pub fused_target_finish: bool,
    pub fused_draft_local: bool,
    pub warp_draft_q: bool,
    pub fused_residual_local: bool,
}

impl Policy {
    pub(crate) fn from_env() -> Result<Self> {
        Self::read(|name| match std::env::var(name) {
            Ok(value) => Ok(Some(value)),
            Err(std::env::VarError::NotPresent) => Ok(None),
            Err(error) => Err(error.into()),
        })
    }
    fn read(mut get: impl FnMut(&str) -> Result<Option<String>>) -> Result<Self> {
        let mut flag = |name: &str, default: bool| -> Result<bool> {
            match get(name)? {
                None => Ok(default),
                Some(value) => {
                    ensure!(value == "0" || value == "1", "{name} must be0or1");
                    Ok(value == "1")
                }
            }
        };
        let policy = Self {
            hierarchical_draft: flag("CANADA_SCRATCH_HIERARCHICAL_DRAFT", false)?,
            local_target: flag("CANADA_SCRATCH_LOCAL_TARGET", false)?,
            local_history: flag("CANADA_SCRATCH_LOCAL_HISTORY", false)?,
            bundled_ids: flag("CANADA_SCRATCH_BUNDLED_SAMPLER", false)?,
            fused_target_controls: flag("CANADA_SCRATCH_FUSED_TARGET_TRANSPORT", false)?,
            fused_target_finish: flag("CANADA_SCRATCH_FUSED_TARGET_FINISH", false)?,
            fused_draft_local: flag("CANADA_SCRATCH_FUSED_DRAFT_LOCAL", false)?,
            warp_draft_q: flag("CANADA_SCRATCH_WARP_DRAFT_Q", false)?,
            fused_residual_local: flag("CANADA_SCRATCH_FUSED_RESIDUAL_LOCAL", false)?,
        };
        ensure!(!policy.local_target || policy.hierarchical_draft,
            "local target transport requires hierarchical drafting");
        ensure!(!policy.local_history || policy.local_target,
            "local history transport requires the prepared local target samplers");
        ensure!(!policy.bundled_ids || policy.local_target,
            "bundled target/residual IDs require local target transport");
        ensure!(!policy.fused_target_controls || policy.local_target,
            "fused target controls require local target transport");
        ensure!(!policy.fused_target_finish || policy.local_target,
            "fused target finish requires local target transport");
        ensure!(!policy.fused_draft_local || policy.hierarchical_draft,
            "fused local draft sampling requires hierarchical drafting");
        ensure!(!policy.warp_draft_q || policy.hierarchical_draft,
            "warp-shared draft quotient requires hierarchical drafting");
        ensure!(!policy.fused_residual_local || policy.bundled_ids,
            "fused local residual sampling requires bundled target/residual IDs");
        if policy.hierarchical_draft {
            ensure!(flag("CANADA_SCRATCH_STOCHASTIC_DRAFT", false)?
                && flag("CANADA_SCRATCH_BLOCK_VERIFY", true)?,
                "hierarchical drafting requires categorical proposals and block verification");
            // These diagnostics explicitly inspect the replicated q or p
            // buffers. Their separate full-vocabulary controls remain usable.
            for name in ["CANADA_SCRATCH_ROOT_FEEDBACK_SCREEN",
                "CANADA_SCRATCH_ENCODER_CACHE_TRACE", "CANADA_SCRATCH_READOUT_BUFFERED_TRACE",
                "CANADA_SCRATCH_DRAFT_REPLAY_TRACE", "CANADA_SCRATCH_ACCEPTANCE_SHADOW",
                "CANADA_SCRATCH_BLOCK_PROGRESS_SCREEN", "CANADA_SCRATCH_BLOCK_DECIDE_ROWS_VALIDATE"] {
                ensure!(!flag(name, false)?, "{name} requires full-vocabulary sampler transport");
            }
        }
        Ok(policy)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(pairs: &[(&str, &str)]) -> Result<Policy> {
        Policy::read(|name| Ok(pairs.iter().find(|(key, _)| *key == name).map(|(_, value)| value.to_string())))
    }
    #[test]
    fn transport_controls_preserve_existing_default() {
        assert_eq!(read(&[]).unwrap(), Policy::default());
        assert!(read(&[("CANADA_SCRATCH_LOCAL_TARGET", "1")]).is_err());
        assert!(read(&[("CANADA_SCRATCH_WARP_DRAFT_Q", "1")]).is_err());
        assert!(read(&[("CANADA_SCRATCH_FUSED_RESIDUAL_LOCAL", "1")]).is_err());
        assert!(read(&[("CANADA_SCRATCH_HIERARCHICAL_DRAFT", "yes")]).is_err());
    }
    #[test]
    fn transport_candidate_requires_actual_q_block_verification() {
        let mut env = vec![("CANADA_SCRATCH_HIERARCHICAL_DRAFT", "1"),
            ("CANADA_SCRATCH_LOCAL_TARGET", "1")];
        assert!(read(&env).is_err());
        env.push(("CANADA_SCRATCH_STOCHASTIC_DRAFT", "1"));
        assert_eq!(read(&env).unwrap(), Policy { hierarchical_draft: true, local_target: true, ..Policy::default() });
        env.push(("CANADA_SCRATCH_BLOCK_VERIFY", "0"));
        assert!(read(&env).is_err());
    }
    #[test]
    fn replicated_probability_diagnostics_require_control_path() {
        assert!(read(&[("CANADA_SCRATCH_HIERARCHICAL_DRAFT", "1"),
            ("CANADA_SCRATCH_STOCHASTIC_DRAFT", "1"),
            ("CANADA_SCRATCH_ACCEPTANCE_SHADOW", "1")]).is_err());
    }
}
