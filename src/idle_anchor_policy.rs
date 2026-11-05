//! One bounded prediction at a completed foreground boundary.
//! This CPU policy does not change model inputs, sampling or admission floors.
#[derive(Clone,Copy,Default,Debug)]
pub(crate) enum Mode { #[default] Off, RecentLargeAppend, EveryEligibleGap }
#[derive(Default)]
pub(crate) struct Policy { mode:Mode,eligible:bool,pending:bool }
impl Policy {
    pub(crate) fn new(enabled:bool)->Self {
        Self::with_mode(if enabled{Mode::RecentLargeAppend}else{Mode::Off})
    }
    pub(crate) fn with_mode(mode:Mode)->Self {Self{mode,..Self::default()}}
    pub(crate) fn began(&mut self,active_reuse:bool,cached:usize,uncached:usize) {
        self.pending=false;
        //128 is the existing encoder-checkpoint append threshold, not a
        // presumed user context range. A previous large forward append is a
        // cheap causal hint; no prompt text or future tokens are inspected.
        self.eligible=match self.mode {
            Mode::Off=>false,
            Mode::RecentLargeAppend=>active_reuse&&cached>=8192&&uncached>=128,
            // Explicit rolling-policy experiment. Actual storage and ingress
            // eligibility are still checked at the idle boundary.
            Mode::EveryEligibleGap=>cached.saturating_add(uncached)>=8192,
        };
    }
    pub(crate) fn finished(&mut self,successful:bool,background:bool) {
        self.pending=self.eligible&&successful&&!background;
        self.eligible=false;
    }
    pub(crate) fn take(&mut self)->bool {std::mem::take(&mut self.pending)}
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn predicts_once_from_past_forward_work_and_never_delays_an_owner() {
        let mut p=Policy::new(true);
        for (active,cached,uncached,success,background,expected) in [
            (false,0,16000,true,false,false),(true,16000,127,true,false,false),
            (true,16000,256,false,false,false),(true,16000,256,true,true,false),
            (true,16000,256,true,false,true)] {
            p.began(active,cached,uncached);assert!(!p.take());
            p.finished(success,background);assert_eq!(p.take(),expected);assert!(!p.take());
        }
        p.began(true,16000,256);p.finished(true,false);
        p.began(true,16256,4);assert!(!p.take());
        let mut off=Policy::new(false);off.began(true,16000,256);off.finished(true,false);assert!(!off.take());
    }
    #[test]
    fn every_gap_control_still_requires_success_and_sufficient_context() {
        let mut p=Policy::with_mode(Mode::EveryEligibleGap);
        for (active,cached,uncached,success,background,expected) in [
            (false,0,8191,true,false,false),(false,0,16000,true,false,true),
            (true,16000,1,true,false,true),(true,16000,256,false,false,false),
            (true,16000,256,true,true,false)] {
            p.began(active,cached,uncached);assert!(!p.take());
            p.finished(success,background);assert_eq!(p.take(),expected);assert!(!p.take());
        }
    }
}
