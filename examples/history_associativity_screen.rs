//! Discrete token-index experiment only. No CPU model or probability math.
//! Compare equal-memory direct/two/four-way buckets and a256KiB strong-key backup.
use anyhow::{Result,ensure};
use serde_json::{Value,json};
use sha2::{Digest,Sha256};
use std::time::Instant;
#[path="../src/history_lookup.rs"] mod history_lookup;
const LENGTHS:[usize;5]=[8,12,16,32,64];
const SLOTS:usize=262144;
const EMPTY:u32=u32::MAX;
fn hash(ids:impl IntoIterator<Item=u32>)->usize {
    let mut v=0xcbf29ce484222325u64;
    for id in ids{v=(v^id as u64).wrapping_mul(0x100000001b3);}
    (v^(v>>32))as usize
}
fn backup_hash(ids:impl IntoIterator<Item=u32>)->usize {
    let mut h=hash(ids)as u64;h^=h>>29;h=h.wrapping_mul(0x9e3779b97f4a7c15);h^=h>>32;h as usize
}
struct Index {tokens:Vec<u32>,tables:[Vec<u32>;5],ways:usize,backup:Vec<u32>}
impl Index {
    fn new(prefix:&[u32],ways:usize,backup:bool)->Self {
        let mut s=Self{tokens:Vec::with_capacity(1048576),tables:std::array::from_fn(|_|vec![EMPTY;SLOTS]),ways,
            backup:if backup{vec![EMPTY;1<<16]}else{Vec::new()}};
        for &v in prefix{s.push(v);}s
    }
    fn push(&mut self,id:u32) {
        self.tokens.push(id);
        let Some(end)=self.tokens.len().checked_sub(6) else{return};
        for (n,table) in LENGTHS.into_iter().zip(&mut self.tables) {
            if end+1<n{continue;}
            let key=&self.tokens[end+1-n..=end];
            let slot=(hash(key.iter().copied())&((SLOTS/self.ways)-1))*self.ways;
            let bucket=&mut table[slot..slot+self.ways];
            if n==64&&!self.backup.is_empty()&&bucket[0]!=EMPTY {
                assert_eq!(self.ways,1);let old=bucket[0]as usize;
                let old_key=&self.tokens[old+1-n..=old];
                if old_key!=key {
                    let b=backup_hash(old_key.iter().copied())&(self.backup.len()-1);
                    self.backup[b]=old as u32;
                }
            }
            // Update an existing key instead of wasting a way on the same
            // phrase. Full token equality, never a hash-only identity claim.
            let victim=bucket.iter().position(|&e|e==EMPTY||{
                let e=e as usize;self.tokens[e+1-n..=e]==*key
            }).unwrap_or(self.ways-1);
            bucket[..=victim].rotate_right(1);bucket[0]=end as u32;
        }
    }
    fn propose(&self,anchor:u32)->Option<(usize,usize,[u32;5])>{
        for(n,table)in LENGTHS.into_iter().zip(&self.tables).rev(){
            if self.tokens.len()+1<n{continue;}
            let suffix=&self.tokens[self.tokens.len()+1-n..];
            let slot=(hash(suffix.iter().copied().chain([anchor]))&((SLOTS/self.ways)-1))*self.ways;
            for &e in &table[slot..slot+self.ways]{
                if e==EMPTY{continue;}let e=e as usize;
                if self.tokens[e]==anchor&&self.tokens[e+1-n..e]==*suffix{
                    return Some((n,e,self.tokens[e+1..e+6].try_into().unwrap()));
                }
            }
            if n==64&&!self.backup.is_empty(){
                let slot=backup_hash(suffix.iter().copied().chain([anchor]))&(self.backup.len()-1);
                let e=self.backup[slot];
                if e!=EMPTY {let e=e as usize;
                    if self.tokens[e]==anchor&&self.tokens[e+1-n..e]==*suffix {
                        return Some((n,e,self.tokens[e+1..e+6].try_into().unwrap()));
                    }
                }
            }
        }
        None
    }
}
fn main()->Result<()> {
    let args:Vec<_>=std::env::args_os().collect();ensure!(args.len()==3,"input and fresh output paths");
    let input:Value=serde_json::from_slice(&std::fs::read(&args[1])?)?;
    ensure!(!std::path::Path::new(&args[2]).exists(),"immutable output");
    let raw=std::fs::read(input["fixture_path"].as_str().unwrap())?;
    ensure!(format!("{:x}",Sha256::digest(&raw))==input["fixture_sha256"].as_str().unwrap(),"fixture pin");
    let fixture:Value=serde_json::from_slice(&raw)?;
    let corpus:Vec<u32>=serde_json::from_value(fixture["tokens"].clone())?;
    let mut cases=Vec::new();
    for c in input["cases"].as_array().unwrap(){
        let n=c["corpus_tokens"].as_u64().unwrap()as usize;
        let mut prefix=corpus[..n].to_vec();prefix.extend(serde_json::from_value::<Vec<u32>>(c["warmup"].clone())?);
        let emitted:Vec<u32>=serde_json::from_value(c["tokens"].clone())?;
        let mut reference=history_lookup::HistoryLookup::with_maximum(1048576,&prefix,SLOTS,64)?;
        let mut canonical=Vec::new();
        for &id in &emitted {canonical.push(reference.propose(reference.position(),id)?.map(|p|(p.matched,p.source_end,p.tokens)));reference.commit(reference.position(),&[id])?;}
        let mut variants=Vec::new();
        for(ways,backup)in [(1,false),(2,false),(4,false),(1,true)]{
            let start=Instant::now();let mut index=Index::new(&prefix,ways,backup);let build_seconds=start.elapsed().as_secs_f64();
            let mut production=history_lookup::HistoryLookup::with_associativity(1048576,&prefix,SLOTS,64,ways)?;
            if backup {production.resize_strong_backup_for_probe(true)?;}
            let start=Instant::now();let(mut proposals,mut long,mut matched,mut changed,mut gained,mut lost)=(0,0,0,0,0,0);
            let mut recovered_offsets=Vec::new();
            for(offset,&anchor)in emitted.iter().enumerate(){
                let p=index.propose(anchor);
                ensure!(production.propose(production.position(),anchor)?.map(|p|(p.matched,p.source_end,p.tokens))==p,
                    "native index differs from independent bucket screen");
                if ways==1&&!backup{ensure!(p==canonical[offset],"direct-map control differs from production index");}
                if backup&&p!=canonical[offset] {ensure!(p.as_ref().is_some_and(|p|p.0==64)
                    &&canonical[offset].as_ref().is_none_or(|p|p.0<64),"backup changed a primary strong or weaker-only decision");
                    if recovered_offsets.len()<4&&recovered_offsets.last().is_none_or(|&last|offset>=last+64){recovered_offsets.push(offset);}}
                if offset+5<emitted.len(){
                    proposals+=usize::from(p.is_some());long+=usize::from(p.as_ref().is_some_and(|p|p.0==64));
                    matched+=p.as_ref().map_or(0,|p|p.2.iter().zip(&emitted[offset+1..offset+6]).take_while(|(a,b)|a==b).count());
                    changed+=usize::from(p!=canonical[offset]);gained+=usize::from(p.is_some()&&canonical[offset].is_none());lost+=usize::from(p.is_none()&&canonical[offset].is_some());
                }
                index.push(anchor);
                production.commit(production.position(),&[anchor])?;
            }
            variants.push(json!({"ways":ways,"strong_backup":backup,"table_bytes":5*SLOTS*4+index.backup.len()*4,"build_seconds":build_seconds,"replay_seconds":start.elapsed().as_secs_f64(),"proposals":proposals,"long64_matches":long,"observed_prefix_matches":matched,"changed":changed,"gained":gained,"lost":lost,"recovered_offsets":recovered_offsets}));
        }
        cases.push(json!({"name":c["name"],"prefix_tokens":prefix.len(),"positions":emitted.len()-5,"variants":variants}));
    }
    std::fs::write(&args[2],serde_json::to_vec_pretty(&json!({"cases":cases,"direct_control_matches_production":true,"scope":"Discrete accepted-token replay on saved public continuations. Direct and associative tables use equal memory; strong-key backup adds256KiB while preserving direct strong matches and all other choices unless it recovers a64-token match. No model execution or numerical oracle. Overlapping match counts are opportunity proxies, not TPS, target probabilities or task quality."}))?)?;Ok(())
}
