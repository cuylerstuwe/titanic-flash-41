//! Head-only scheduling experiment. It never changes slots or model state.
use super::*;

pub(super) struct Turns {
    pub(super) enabled: bool,
    pub(super) decode_prefill_slice: Option<Duration>,
    handoff_wait: Duration,
    handoff_pending: bool,
    owner: Option<u64>,
    owner_http: Option<String>,
    first_until: Option<Instant>,
    owed: Option<String>,
    protected_until: Option<Instant>,
    slice: Duration,
}
impl Turns {
    #[cfg(test)]
    pub(super) fn for_test()->Self {Self{enabled:true,decode_prefill_slice:None,handoff_wait:Duration::from_millis(100),handoff_pending:false,owner:None,owner_http:None,first_until:None,owed:None,protected_until:None,slice:Duration::from_secs(12)}}
    pub(super) fn new(slice:Duration)->Result<Self> {
        let enabled=match std::env::var("CANADA_SCRATCH_WORK_AWARE_PREFILL") {
            Ok(v)=>{ensure!(v=="0"||v=="1","work-aware prefill flag");v=="1"},
            Err(std::env::VarError::NotPresent)=>false,
            Err(e)=>return Err(e.into()),
        };
        let decode_prefill_slice=match std::env::var("CANADA_SCRATCH_WORK_AWARE_DECODE_PREFILL_MS") {
            Ok(v)=>{
                let ms=v.parse::<u64>()?;
                ensure!(ms==0||(enabled&&(250..=12000).contains(&ms)),"work-aware decoder prefill slice");
                (ms>0).then(||Duration::from_millis(ms))
            },
            Err(std::env::VarError::NotPresent)=>None,
            Err(e)=>return Err(e.into()),
        };
        let handoff_wait=match std::env::var("CANADA_SCRATCH_WORK_AWARE_HANDOFF_WAIT_MS") {
            Ok(v)=>{
                let ms=v.parse::<u64>()?;
                ensure!(ms==0||(enabled&&ms<=100),"work-aware handoff wait");
                Duration::from_millis(ms)
            },
            Err(std::env::VarError::NotPresent)=>Duration::ZERO,
            Err(e)=>return Err(e.into()),
        };
        Ok(Self{enabled,decode_prefill_slice,handoff_wait,handoff_pending:false,owner:None,owner_http:None,first_until:None,owed:None,protected_until:None,slice})
    }
    pub(super) fn retired(&mut self) {self.handoff_pending=self.enabled;}
    pub(super) fn take_handoff_wait(&mut self)->Duration {
        if std::mem::take(&mut self.handoff_pending)&&self.enabled {self.handoff_wait}else{Duration::ZERO}
    }
    pub(super) fn refresh(&mut self,current:Option<&str>,ready:&VecDeque<(Option<u64>,Job)>,
        prefill:bool,emitted:usize) {
        if self.owed.as_deref().is_some_and(|id|Some(id)!=current&&!ready.iter().any(|(_,job)|job.id==id)) {
            self.owed=None;self.protected_until=None;
        }
        if self.owed.as_deref()==current&&current.is_some()
            &&((!prefill&&emitted>=QUANTUM)||self.protected_until.is_some_and(|end|Instant::now()>=end)) {
            self.owed=None;self.protected_until=None;
        }
    }
    pub(super) fn activate(&mut self,id:u64,http:&str,fresh:bool,peer:Option<(&str,bool)>) {
        if !self.enabled||self.owner==Some(id){return;}
        let now=Instant::now();self.owner=Some(id);self.owner_http=Some(http.into());
        // A long first generation turn helps only when the other owner is
        // still prefilling. Two decoders retain their usual64-token turns.
        self.first_until=(fresh&&peer.is_some_and(|(_,prefill)|prefill)).then_some(now+self.slice);
        if fresh&&self.owed.is_none(){self.owed=peer.map(|(http,_)|http.into());}
        self.protected_until=(self.owed.as_deref()==Some(http)).then_some(now+self.slice);
    }
    pub(super) fn first_grant(&self)->bool {
        self.enabled&&self.first_until.is_some_and(|end|Instant::now()<end)
    }
    pub(super) fn owes(&self,http:&str)->bool {self.enabled&&self.owed.as_deref()==Some(http)}
    pub(super) fn protected(&self,prefill:bool,emitted:usize)->bool {
        self.enabled&&self.owner.is_some()&&self.owner_http==self.owed
            &&if prefill{self.protected_until.is_some_and(|end|Instant::now()<end)}else{emitted<QUANTUM}
    }
    pub(super) fn prefill_protection(&self)->Option<Instant> {
        if self.enabled&&self.owner_http==self.owed{self.protected_until}else{None}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn handoff_wait_is_once_per_terminal_vacancy_and_disabled_without_policy() {
        let mut p=Turns::for_test();
        assert!(p.take_handoff_wait().is_zero());
        p.retired();assert_eq!(p.take_handoff_wait(),Duration::from_millis(100));
        assert!(p.take_handoff_wait().is_zero());
        p.retired();p.enabled=false;assert!(p.take_handoff_wait().is_zero());
        p.enabled=true;assert!(p.take_handoff_wait().is_zero());
    }
    #[test]
    fn newcomer_grant_is_specific_to_a_prefilling_peer_and_repaid() {
        let mut p=Turns::for_test();p.activate(2,"short",true,Some(("long",true)));
        assert!(p.first_grant());assert!(p.owes("long"));assert!(!p.protected(true,0));
        p.activate(1,"long",false,None);assert!(!p.first_grant());assert!(p.protected(true,0));
        assert!(p.protected(false,63));assert!(!p.protected(false,64));
        p.refresh(Some("long"),&VecDeque::new(),false,64);assert!(!p.owes("long"));
        p.activate(3,"next",true,Some(("long",false)));assert!(!p.first_grant());assert!(p.owes("long"));
    }
    #[test]
    fn repeated_selection_does_not_restart_a_slice_and_terminal_owners_leave() {
        let mut p=Turns::for_test();p.activate(2,"short",true,Some(("long",true)));let until=p.first_until;
        p.activate(2,"short",true,Some(("long",true)));assert_eq!(p.first_until,until);
        p.refresh(Some("short"),&VecDeque::new(),false,0);assert!(!p.owes("long"));
    }
    #[test]
    fn an_unstarted_owner_keeps_its_debt_when_it_receives_a_model_id() {
        let mut p=Turns::for_test();p.activate(1,"short",true,Some(("long",true)));
        assert!(p.owes("long"));
        p.activate(2,"long",true,Some(("another",true)));
        assert!(p.owes("long"));assert!(p.protected(true,0));
        p.protected_until=Some(Instant::now());
        p.refresh(Some("long"),&VecDeque::new(),true,0);assert!(!p.owes("long"));
    }
}
