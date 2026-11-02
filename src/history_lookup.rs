//! Bounded exact-token history proposals. This is discrete host scheduling,
//! never model arithmetic: all probabilities and target verification stay GPU.
//!
//! A proposal is a deterministic point mass conditional on already committed
//! text and the pending emitted anchor.
use anyhow::{Result, ensure};

const LENGTHS: [usize; 5] = [8, 12, 16, 32, 64];
const SLOTS: usize = 1 << 16;
const MAX_SLOTS: usize = 1 << 20;
const EMPTY: u32 = u32::MAX;
pub(crate) const STRONG_BACKUP_BYTES: usize = 65536 * 4;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Proposal {
    pub tokens: [u32; 5],
    pub matched: usize,
    pub source_end: usize,
}

#[derive(Clone)]
pub(crate) struct HistoryLookup {
    tokens: Vec<u32>,
    tables: [Vec<u32>; 5],
    active_lengths: usize,
    capacity: usize,
    minimum_match: Option<usize>,
    ways: usize,
    strong_backup: Vec<u32>,
}

fn fingerprint(ids: impl IntoIterator<Item = u32>) -> u64 {
    // Fixed arithmetic on public token IDs gives the same table slot on every
    // rank. Collisions only replace an optimization opportunity; full token
    // comparison below prevents a hash collision from being called a match.
    let mut value = 0xcbf29ce484222325u64;
    for id in ids {
        value = (value ^ id as u64).wrapping_mul(0x100000001b3);
    }
    value ^ (value >> 32)
}
fn backup_fingerprint(ids:impl IntoIterator<Item=u32>)->usize {
    let mut h=fingerprint(ids);h^=h>>29;h=h.wrapping_mul(0x9e3779b97f4a7c15);h^=h>>32;h as usize
}

impl HistoryLookup {
    pub(crate) fn new(capacity: usize, prefix: &[u32]) -> Result<Self> {
        Self::with_slots(capacity, prefix, SLOTS)
    }
    pub(crate) fn with_slots(capacity: usize, prefix: &[u32], slots: usize) -> Result<Self> {
        Self::with_maximum(capacity,prefix,slots,16)
    }
    pub(crate) fn with_maximum(capacity:usize,prefix:&[u32],slots:usize,maximum:usize)->Result<Self> {
        Self::with_associativity(capacity,prefix,slots,maximum,1)
    }
    pub(crate) fn with_strong_backup(capacity:usize,prefix:&[u32],slots:usize)->Result<Self> {
        let mut result=Self::with_maximum(capacity,&[],slots,64)?;
        result.strong_backup=vec![EMPTY;STRONG_BACKUP_BYTES/4];
        if !prefix.is_empty(){result.commit(0,prefix)?;}
        Ok(result)
    }
    pub(crate) fn with_associativity(capacity:usize,prefix:&[u32],slots:usize,maximum:usize,ways:usize)->Result<Self> {
        ensure!((1..=1048576).contains(&capacity) && slots.is_power_of_two() && slots <= MAX_SLOTS,
            "history lookup capacity");
        ensure!([1,2,4].contains(&ways)&&slots>=ways,"history bucket geometry");
        ensure!([16,32,64].contains(&maximum),"history maximum match");
        let active_lengths=LENGTHS.iter().position(|&n|n==maximum).unwrap()+1;
        let mut result = Self {
            tokens: Vec::with_capacity(capacity),
            tables: std::array::from_fn(|i|if i<active_lengths{vec![EMPTY; slots]}else{Vec::new()}),
            active_lengths,
            capacity,
            minimum_match: Some(8),
            ways,
            strong_backup: Vec::new(),
        };
        if !prefix.is_empty() {
            result.commit(0, prefix)?;
        }
        Ok(result)
    }
    /// Preserve reserved token capacity when a cache image will be extended.
    /// Vec::clone alone shrinks it to the current length and hides later growth
    /// from the shared snapshot budget.
    pub(crate) fn clone_reserved(&self) -> Self {
        let mut tokens=Vec::with_capacity(self.capacity);
        tokens.extend_from_slice(&self.tokens);
        Self{tokens,tables:self.tables.clone(),active_lengths:self.active_lengths,
            capacity:self.capacity,minimum_match:self.minimum_match,ways:self.ways,
            strong_backup:self.strong_backup.clone()}
    }
    pub(crate) fn position(&self) -> u64 {
        self.tokens.len() as u64
    }
    pub(crate) fn state_digest(&self) -> String {
        use sha2::{Digest, Sha256};
        let mut hash = Sha256::new();
        hash.update((self.capacity as u64).to_le_bytes());
        hash.update((self.minimum_match.unwrap_or(0) as u64).to_le_bytes());
        hash.update((self.tokens.len() as u64).to_le_bytes());
        // Preserve the default16-token index digest and version only the
        // additional accepted-history tables used by this isolated probe.
        if self.active_lengths>3 {
            hash.update(b"extended-history-match-v1\0");
            hash.update((LENGTHS[self.active_lengths-1]as u64).to_le_bytes());
        }
        if self.ways>1 {hash.update(b"associative-history-v1\0");hash.update((self.ways as u64).to_le_bytes());}
        if !self.strong_backup.is_empty() {
            hash.update(b"strong-history-backup-v1\0");
            for &value in &self.strong_backup {hash.update(value.to_le_bytes());}
        }
        for &v in self.tokens.iter().chain(self.tables.iter().flatten()) {
            hash.update(v.to_le_bytes());
        }
        format!("{:x}", hash.finalize())
    }
    pub(crate) fn workspace_bytes(&self) -> usize {
        (self.tokens.capacity() + self.tables.iter().map(|t| t.capacity()).sum::<usize>()+self.strong_backup.capacity())
            * std::mem::size_of::<u32>()
    }
    pub(crate) fn reserved_workspace_bytes(&self)->usize {
        (self.capacity+self.tables.iter().map(|t|t.capacity()).sum::<usize>()+self.strong_backup.capacity())
            *std::mem::size_of::<u32>()
    }
    /// Choose the proposal source from already accepted history only. This
    /// never inspects future draft draws or target verification probabilities.
    /// Keep indexing while suppressed, so changing policy does not rebuild or
    /// alter the accepted-prefix state; snapshots retain this policy too.
    pub(crate) fn set_minimum_match(&mut self, minimum: Option<usize>) -> Result<()> {
        ensure!(minimum.is_none_or(|n| LENGTHS[..self.active_lengths].contains(&n)), "history minimum match");
        self.minimum_match = minimum;
        Ok(())
    }
    /// Rebuild only the discrete accepted-token index. A larger table retains
    /// more collision-displaced candidates; full token matching still governs.
    pub(crate) fn resize_tables_for_probe(&mut self, slots: usize) -> Result<()> {
        ensure!(self.strong_backup.is_empty(),"independent history geometry probes");
        ensure!([1<<16,1<<18,1<<20].contains(&slots), "history table probe size");
        let mut replacement=Self::with_associativity(self.capacity,&self.tokens,slots,LENGTHS[self.active_lengths-1],self.ways)?;
        replacement.minimum_match=self.minimum_match;
        *self=replacement;
        Ok(())
    }
    pub(crate) fn resize_maximum_for_probe(&mut self,maximum:usize)->Result<()> {
        ensure!(self.strong_backup.is_empty(),"independent history geometry probes");
        ensure!(self.minimum_match.is_none_or(|n|n<=maximum),"maximum below active minimum");
        let mut replacement=Self::with_associativity(self.capacity,&self.tokens,self.tables[0].len(),maximum,self.ways)?;
        replacement.minimum_match=self.minimum_match;*self=replacement;Ok(())
    }
    pub(crate) fn resize_associativity_for_probe(&mut self,ways:usize)->Result<()> {
        ensure!(self.strong_backup.is_empty(),"independent history geometry probes");
        let mut replacement=Self::with_associativity(self.capacity,&self.tokens,self.tables[0].len(),LENGTHS[self.active_lengths-1],ways)?;
        replacement.minimum_match=self.minimum_match;*self=replacement;Ok(())
    }
    pub(crate) fn resize_strong_backup_for_probe(&mut self,enabled:bool)->Result<()> {
        ensure!(self.ways==1&&LENGTHS[self.active_lengths-1]==64,"strong backup requires direct64 index");
        let mut replacement=if enabled{Self::with_strong_backup(self.capacity,&self.tokens,self.tables[0].len())?}
            else{Self::with_maximum(self.capacity,&self.tokens,self.tables[0].len(),64)?};
        replacement.minimum_match=self.minimum_match;*self=replacement;Ok(())
    }
    /// Publish only actual committed inputs, including the old pending anchor.
    /// The next emitted token stays outside this history just as it stays out
    /// of target KV. Admission errors leave the lookup unchanged.
    pub(crate) fn commit(&mut self, start: u64, accepted: &[u32]) -> Result<()> {
        ensure!(start == self.position() && !accepted.is_empty()
            && accepted.iter().all(|&id| id < 129280)
            && self.tokens.len().checked_add(accepted.len()).is_some_and(|n| n <= self.capacity),
            "history lookup accepted-prefix/frontier admission");
        for &id in accepted {
            self.tokens.push(id);
            // Exactly five continuation tokens are now already committed.
            // Never index the pending anchor or an unverified proposal suffix.
            if let Some(end) = self.tokens.len().checked_sub(6) {
                for (n, table) in LENGTHS.into_iter().zip(&mut self.tables).take(self.active_lengths) {
                    if end + 1 >= n {
                        let key = &self.tokens[end + 1 - n..=end];
                        let slot = (fingerprint(key.iter().copied()) as usize & (table.len()/self.ways - 1))*self.ways;
                        if n==64&&!self.strong_backup.is_empty()&&table[slot]!=EMPTY {
                            let old=table[slot]as usize;let old_key=&self.tokens[old+1-n..=old];
                            if old_key!=key {
                                let b=backup_fingerprint(old_key.iter().copied())&(self.strong_backup.len()-1);
                                self.strong_backup[b]=old as u32;
                            }
                        }
                        if self.ways==1 {table[slot]=end as u32;}
                        else {
                            let bucket=&mut table[slot..slot+self.ways];
                            // Preserve the most recent occurrence of each full
                            // key; duplicate phrases must not consume ways.
                            let victim=bucket.iter().position(|&e|e==EMPTY||{
                                let e=e as usize;self.tokens[e+1-n..=e]==*key
                            }).unwrap_or(self.ways-1);
                            bucket[..=victim].rotate_right(1);bucket[0]=end as u32;
                        }
                    }
                }
            }
        }
        Ok(())
    }
    /// Extend only a64-token match with15 already committed source tokens.
    /// No future target draw or likelihood participates in proposal selection.
    pub(crate) fn propose_long(&self, start:u64, anchor:u32)->Result<Option<(Proposal,[u32;15])>> {
        Ok(self.propose_block::<16>(start,anchor)?.map(|(source,inputs)|
            (source,inputs[1..].try_into().unwrap())))
    }
    /// A whole block is chosen from committed history before target sampling.
    /// The anchor is included so the caller's const geometry also covers the
    /// final bonus row, without an allocation or an unchecked trailing suffix.
    pub(crate) fn propose_block<const ROWS:usize>(&self,start:u64,anchor:u32)
        ->Result<Option<(Proposal,[u32;ROWS])>> {
        self.propose_block_minimum::<ROWS>(start,anchor,64)
    }
    /// The width threshold is a proposal policy, not an acceptance shortcut.
    /// Every selected token must still come from committed source storage.
    pub(crate) fn propose_block_minimum<const ROWS:usize>(&self,start:u64,anchor:u32,minimum:usize)
        ->Result<Option<(Proposal,[u32;ROWS])>> {
        ensure!([16,32].contains(&ROWS)&&[16,24,32,48,64].contains(&minimum),"history block geometry/minimum");
        let Some(mut proposal)=self.propose(start,anchor)? else{return Ok(None)};
        if proposal.source_end+ROWS>self.tokens.len() {return Ok(None)}
        if proposal.matched<minimum {
            // Optional intermediate thresholds use the existing shorter table
            // and prove the extra *past* tokens literally. No additional table
            // or future proposal/target draw is needed. Preserve the original
            // indexed32/64 decisions, including their collision behavior.
            let required_key=match minimum{24=>16,48=>32,_=>return Ok(None)};
            if proposal.matched<required_key||proposal.source_end+1<minimum||self.tokens.len()+1<minimum
                ||self.tokens[proposal.source_end+1-minimum..proposal.source_end]
                    !=self.tokens[self.tokens.len()+1-minimum..] {return Ok(None)}
            proposal.matched=minimum;
        }
        let inputs=self.tokens[proposal.source_end..proposal.source_end+ROWS].try_into().unwrap();
        Ok(Some((proposal,inputs)))
    }
    pub(crate) fn propose(&self, start: u64, anchor: u32) -> Result<Option<Proposal>> {
        ensure!(start == self.position() && anchor < 129280, "history lookup query frontier/token");
        let Some(minimum) = self.minimum_match else { return Ok(None); };
        for (n, table) in LENGTHS.into_iter().zip(&self.tables).take(self.active_lengths).rev() {
            if n < minimum || self.tokens.len() + 1 < n {
                continue;
            }
            let suffix = &self.tokens[self.tokens.len() + 1 - n..];
            let slot = (fingerprint(suffix.iter().copied().chain([anchor])) as usize & (table.len()/self.ways - 1))*self.ways;
            for &end in &table[slot..slot+self.ways] {
            if end == EMPTY {
                continue;
            }
            let end = end as usize;
            ensure!(end + 1 >= n && end + 6 <= self.tokens.len(), "history lookup index bounds");
            if self.tokens[end] != anchor || self.tokens[end + 1 - n..end] != *suffix {
                continue;
            }
            return Ok(Some(Proposal {
                tokens: self.tokens[end + 1..end + 6].try_into().unwrap(),
                matched: n,
                source_end: end,
            }));
            }
            // Consult this only after the normal64-token key misses. Every
            // original strong hit and every shorter-only choice is preserved
            // unless a displaced, fully token-checked64-token key is recovered.
            if n==64&&!self.strong_backup.is_empty() {
                let slot=backup_fingerprint(suffix.iter().copied().chain([anchor]))&(self.strong_backup.len()-1);
                let end=self.strong_backup[slot];
                if end!=EMPTY {let end=end as usize;
                    ensure!(end+1>=64&&end+6<=self.tokens.len(),"backup history index bounds");
                    if self.tokens[end]==anchor&&self.tokens[end+1-n..end]==*suffix {
                        return Ok(Some(Proposal{tokens:self.tokens[end+1..end+6].try_into().unwrap(),matched:64,source_end:end}));
                    }
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_block_gate_keeps_source_committed_and_default_strong() {
        let mut prefix=(100..132).collect::<Vec<_>>();prefix.extend(300..350);prefix.extend(100..131);
        let lookup=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
        assert!(lookup.propose_block::<32>(lookup.position(),131).unwrap().is_none());
        let (proposal,ids)=lookup.propose_block_minimum::<32>(lookup.position(),131,32).unwrap().unwrap();
        assert_eq!(proposal.matched,32);assert_eq!(ids[0],131);
        assert_eq!(ids[1..],(300..331).collect::<Vec<_>>()[..]);
        let periodic=(0..96).map(|i|100+i%8).collect::<Vec<_>>();
        let recent=HistoryLookup::with_maximum(256,&periodic,SLOTS,64).unwrap();
        assert!(recent.propose_block_minimum::<16>(recent.position(),100,32).unwrap().is_none());
        assert!(lookup.propose_block_minimum::<32>(lookup.position(),131,12).is_err());
    }

    #[test]
    fn intermediate_block_gate_proves_the_extra_past_prefix() {
        for proof in [24usize,48] {
            let key=if proof==24{16}else{32};let anchor=99+proof as u32;
            let mut prefix=(100..=anchor).collect::<Vec<_>>();prefix.extend(300..350);prefix.extend(100..anchor);
            let lookup=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
            assert_eq!(lookup.propose(lookup.position(),anchor).unwrap().unwrap().matched,key);
            let (p,_)=lookup.propose_block_minimum::<16>(lookup.position(),anchor,proof).unwrap().unwrap();
            assert_eq!(p.matched,proof);
            assert!(lookup.propose_block_minimum::<16>(lookup.position(),anchor,key*2).unwrap().is_none());
            let n=prefix.len();prefix[n+1-proof]=999;
            let changed=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
            assert!(changed.propose_block_minimum::<16>(changed.position(),anchor,key).unwrap().is_some());
            assert!(changed.propose_block_minimum::<16>(changed.position(),anchor,proof).unwrap().is_none());
        }
    }

    #[test]
    fn wide_history_block_has_only_committed_tokens_and_keeps_narrow_source() {
        let mut prefix=(100..164).collect::<Vec<_>>();prefix.extend(300..350);prefix.extend(100..163);
        let lookup=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
        let (narrow,a)=lookup.propose_block::<16>(lookup.position(),163).unwrap().unwrap();
        let (wide,b)=lookup.propose_block::<32>(lookup.position(),163).unwrap().unwrap();
        assert_eq!(narrow,wide);assert_eq!(a,b[..16]);assert_eq!(b[0],163);
        assert_eq!(b[1..],(300..331).collect::<Vec<_>>()[..]);
        // A recent repeated key can have enough source continuation for M16
        // while M32 would run into the pending, uncommitted part of the request.
        let periodic=(0..128).map(|i|100+i%16).collect::<Vec<_>>();
        let overlap=HistoryLookup::with_maximum(256,&periodic,SLOTS,64).unwrap();
        assert!(overlap.propose_block::<16>(overlap.position(),100).unwrap().is_some());
        assert!(overlap.propose_block::<32>(overlap.position(),100).unwrap().is_none());
        assert!(lookup.propose_block::<32>(lookup.position()+1,163).is_err());
        assert!(lookup.propose_block::<8>(lookup.position(),163).is_err());
    }

    #[test]
    fn long_proposal_requires_strong_match_and_committed_continuation() {
        let mut prefix=(100..164).collect::<Vec<_>>();prefix.extend(300..330);prefix.extend(100..163);
        let lookup=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
        let (source,tokens)=lookup.propose_long(lookup.position(),163).unwrap().unwrap();
        assert_eq!(source.matched,64);assert_eq!(tokens,(300..315).collect::<Vec<_>>()[..]);
        let short=HistoryLookup::new(256,&prefix).unwrap();
        assert!(short.propose_long(short.position(),163).unwrap().is_none());
        let mut brief=(100..164).collect::<Vec<_>>();brief.extend(300..305);brief.extend(100..163);
        let lookup=HistoryLookup::with_maximum(256,&brief,SLOTS,64).unwrap();
        // The source has enough committed continuation, but it includes a new
        // repetition after the five payload tokens; selecting it remains causal.
        let (_,tokens)=lookup.propose_long(lookup.position(),163).unwrap().unwrap();
        assert_eq!(&tokens[..5],&[300,301,302,303,304]);assert_eq!(tokens[5],100);
        assert!(lookup.propose_long(lookup.position()+1,163).is_err());
        let overlap=HistoryLookup::with_maximum(256,&vec![7;100],SLOTS,64).unwrap();
        assert!(overlap.propose_long(overlap.position(),7).unwrap().is_none());
    }
    #[test]
    fn longer_history_distinguishes_continuations_and_preserves_accepted_snapshots() {
        let a:Vec<u32>=(100..164).collect();
        let mut prefix=a.clone();prefix.extend(200..205);prefix.push(999);
        prefix.extend(1000..1048);prefix.extend_from_slice(&a[48..]);
        prefix.extend(300..305);prefix.push(998);prefix.extend_from_slice(&a[..63]);
        let mut lookup=HistoryLookup::new(256,&prefix).unwrap();let pos=lookup.position();
        assert_eq!(lookup.propose(pos,163).unwrap().unwrap().tokens,[300,301,302,303,304]);
        let original=lookup.state_digest();let bytes=lookup.workspace_bytes();
        lookup.resize_maximum_for_probe(64).unwrap();
        let proposal=lookup.propose(pos,163).unwrap().unwrap();
        assert_eq!((proposal.matched,proposal.tokens),(64,[200,201,202,203,204]));
        assert_eq!(lookup.workspace_bytes(),bytes+2*SLOTS*4);
        assert!(lookup.resize_maximum_for_probe(33).is_err());
        assert_eq!(lookup.propose(pos,163).unwrap().unwrap(),proposal);
        lookup.resize_maximum_for_probe(16).unwrap();assert_eq!(lookup.state_digest(),original);
        lookup.resize_maximum_for_probe(64).unwrap();let saved=lookup.clone();
        lookup.commit(pos,&[163,200]).unwrap();prefix.extend([163,200]);
        let fresh=HistoryLookup::with_maximum(256,&prefix,SLOTS,64).unwrap();
        assert_eq!(lookup.state_digest(),fresh.state_digest());
        assert_eq!(saved.position(),pos);assert_eq!(saved.propose(pos,163).unwrap().unwrap(),proposal);
    }

    #[test]
    fn table_resize_rebuilds_only_accepted_tokens_and_roundtrips_exactly() {
        let mut tokens:Vec<u32>=(0..1000).map(|i|((i/41+i%13)%37)as u32).collect();
        tokens.extend(200..216);tokens.extend(400..405);tokens.extend(200..215);
        let mut lookup=HistoryLookup::new(2048,&tokens).unwrap();
        lookup.set_minimum_match(Some(12)).unwrap();
        let old=lookup.state_digest();let position=lookup.position();
        assert!(lookup.resize_tables_for_probe(1<<21).is_err());
        assert_eq!(lookup.state_digest(),old);
        lookup.resize_tables_for_probe(1<<18).unwrap();
        let mut fresh=HistoryLookup::with_slots(2048,&tokens,1<<18).unwrap();
        fresh.set_minimum_match(Some(12)).unwrap();
        assert_eq!(lookup.state_digest(),fresh.state_digest());
        assert_eq!(lookup.position(),position);
        assert_eq!(lookup.propose(position,215).unwrap(),fresh.propose(position,215).unwrap());
        let snapshot=lookup.clone();lookup.commit(position,&[215,400]).unwrap();
        assert_eq!(snapshot.state_digest(),fresh.state_digest());
        let mut snapshot=snapshot;snapshot.resize_tables_for_probe(1<<16).unwrap();
        assert_eq!(snapshot.state_digest(),old);
    }

    #[test]
    fn proposal_policy_filters_only_known_matches_and_survives_snapshot_clone() {
        // The only repeated suffix is exactly eight tokens long.
        let mut prefix: Vec<u32> = (100..108).collect();
        prefix.extend(200..205);
        prefix.push(999);
        prefix.extend(100..107);
        let mut lookup = HistoryLookup::new(128, &prefix).unwrap();
        let expected = lookup.propose(prefix.len() as u64, 107).unwrap().unwrap();
        assert_eq!(expected.matched, 8);
        let tables = lookup.tables.clone();
        let digest = lookup.state_digest();
        lookup.set_minimum_match(Some(12)).unwrap();
        assert!(lookup.propose(prefix.len() as u64, 107).unwrap().is_none());
        assert_ne!(lookup.state_digest(), digest);
        let saved = lookup.clone();
        lookup.set_minimum_match(None).unwrap();
        assert!(lookup.propose(prefix.len() as u64, 107).unwrap().is_none());
        assert!(lookup.set_minimum_match(Some(10)).is_err());
        assert_eq!(saved.minimum_match, Some(12));
        assert_eq!(lookup.tables, tables);
        lookup.set_minimum_match(Some(8)).unwrap();
        assert_eq!(lookup.state_digest(), digest);
        assert_eq!(lookup.propose(prefix.len() as u64, 107).unwrap(), Some(expected));
    }

    #[test]
    fn lookup_uses_known_continuations_and_does_not_publish_predictions() {
        let mut prefix: Vec<u32> = (100..116).collect();
        prefix.extend(200..205);
        prefix.extend(100..115);
        let mut lookup = HistoryLookup::new(128, &prefix).unwrap();
        let proposal = lookup.propose(prefix.len() as u64, 115).unwrap().unwrap();
        assert_eq!(proposal, Proposal { tokens: [200,201,202,203,204], matched:16, source_end:15 });
        assert_eq!(lookup.tokens, prefix);
        // A rejected suffix never appears in the accepted host token state.
        lookup.commit(prefix.len() as u64, &[115,200,201]).unwrap();
        prefix.extend([115,200,201]);
        assert_eq!(lookup.tokens, prefix);
        assert_eq!(lookup.position(), prefix.len() as u64);
        assert!(lookup.commit(0, &[3]).is_err());
        assert!(lookup.propose(0, 3).is_err());
    }

    #[test]
    fn checkpoint_clone_preserves_collisions_policy_and_reserved_growth() {
        let prefix:Vec<u32>=(0..300).map(|i|100+i%37).collect();
        let mut original=HistoryLookup::with_slots(1024,&prefix,4).unwrap();
        original.set_minimum_match(Some(12)).unwrap();
        let mut restored=original.clone_reserved();
        assert_eq!(restored.workspace_bytes(),original.workspace_bytes());
        let immutable=restored.clone_reserved();
        let suffix:Vec<u32>=(0..500).map(|i|100+i%31).collect();
        original.commit(300,&suffix).unwrap();restored.commit(300,&suffix).unwrap();
        assert_eq!(restored.state_digest(),original.state_digest());
        assert_eq!(restored.workspace_bytes(),immutable.workspace_bytes());
        assert_eq!(immutable.position(),300);
    }

    #[test]
    fn short_image_clone_readmits_full_capacity_before_continuation() {
        let prefix:Vec<u32>=(0..501).map(|i|100+i%37).collect();
        let original=HistoryLookup::with_associativity(1001,&prefix,4,64,2).unwrap();
        let short=original.clone();
        assert!(short.workspace_bytes()<short.reserved_workspace_bytes());
        let mut restored=short.clone_reserved();
        assert_eq!(restored.workspace_bytes(),short.reserved_workspace_bytes());
        assert_eq!(restored.state_digest(),original.state_digest());
        let bytes=restored.workspace_bytes();
        restored.commit(501,&vec![117;500]).unwrap();
        assert_eq!(restored.workspace_bytes(),bytes);
        assert_eq!(short.position(),501);
    }

    #[test]
    fn strong_backup_keeps_primary_tables_and_survives_snapshot_clone() {
        let mut prefix=(100..700).collect::<Vec<_>>();prefix.extend(100..163);
        let direct=HistoryLookup::with_maximum(1024,&prefix,4,64).unwrap();
        let mut backup=direct.clone_reserved();backup.resize_strong_backup_for_probe(true).unwrap();
        assert_eq!(backup.tables,direct.tables);
        assert_eq!(backup.workspace_bytes(),direct.workspace_bytes()+65536*4);
        let proposal=backup.propose(663,163).unwrap().unwrap();
        assert_eq!(proposal.matched,64);assert_eq!(proposal.tokens,[164,165,166,167,168]);
        let mut saved=backup.clone_reserved();assert_eq!(saved.state_digest(),backup.state_digest());
        saved.commit(663,&[163,164,165]).unwrap();
        assert_eq!(backup.position(),663);assert_eq!(saved.workspace_bytes(),backup.workspace_bytes());
        backup.resize_strong_backup_for_probe(false).unwrap();assert_eq!(backup.state_digest(),direct.state_digest());
    }

    #[test]
    fn invalid_commit_is_atomic_and_memory_is_bounded() {
        let mut lookup = HistoryLookup::new(8, &[10,11,12]).unwrap();
        assert!(lookup.commit(3, &[13,129280]).is_err());
        assert!(lookup.commit(3, &[13;6]).is_err());
        assert!(lookup.commit(3, &[]).is_err());
        assert_eq!(lookup.tokens, [10,11,12]);
        assert!(lookup.propose(3,129280).is_err());
        assert_eq!(lookup.workspace_bytes(), (8+3*SLOTS)*4);
        assert!(HistoryLookup::new(1048577, &[]).is_err());
    }

    #[test]
    fn collisions_are_verified_and_new_sessions_have_no_old_matches() {
        let mut prefix:Vec<u32>=(100..116).collect();
        prefix.extend(200..205);
        prefix.extend(100..115);
        let mut lookup=HistoryLookup::with_slots(128,&prefix,1).unwrap();
        // Force every candidate to another valid but different indexed span.
        for (n,table) in LENGTHS.into_iter().zip(&mut lookup.tables).take(lookup.active_lengths) {
            table[0]=(n+2)as u32;
        }
        assert!(lookup.propose(prefix.len()as u64,115).unwrap().is_none());
        let fresh=HistoryLookup::new(128,&(100..115).collect::<Vec<_>>()).unwrap();
        assert!(fresh.propose(15,115).unwrap().is_none());
    }

    #[test]
    fn incremental_commits_and_one_prefill_build_identical_causal_indexes() {
        let tokens:Vec<u32>=(0..800).map(|i| ((i/41+i%13)%37)as u32).collect();
        let mut incremental=HistoryLookup::new(1024,&[]).unwrap();
        for chunk in tokens.chunks(17) {
            incremental.commit(incremental.position(),chunk).unwrap();
            let fresh=HistoryLookup::new(1024,&tokens[..incremental.position()as usize]).unwrap();
            assert_eq!(incremental.tables,fresh.tables);
            for anchor in 0..37 {
                assert_eq!(incremental.propose(incremental.position(),anchor).unwrap(),
                    fresh.propose(fresh.position(),anchor).unwrap());
            }
        }
    }
}
