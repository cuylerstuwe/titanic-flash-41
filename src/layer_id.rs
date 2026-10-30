//! Canonical checkpoint namespaces and disjoint resident expert identities.
use anyhow::{Result, ensure};
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerId {
    Target(usize),
    Draft(usize),
}
impl LayerId {
    pub fn target(layer: usize) -> Result<Self> {
        ensure!(layer < 40, "target layer range");
        Ok(Self::Target(layer))
    }
    pub fn draft(stage: usize) -> Result<Self> {
        ensure!(stage < 3, "DSpark stage range");
        Ok(Self::Draft(stage))
    }
    pub fn from_key(key: usize) -> Result<Self> {
        if key < 40 {
            Self::target(key)
        } else {
            Self::draft(key - 40)
        }
    }
    pub fn key(self) -> usize {
        match self {
            Self::Target(i) => i,
            Self::Draft(i) => 40 + i,
        }
    }
    pub fn prefix(self) -> String {
        match self {
            Self::Target(i) => format!("layers.{i}"),
            Self::Draft(i) => format!("mtp.{i}"),
        }
    }
    pub fn is_draft(self) -> bool {
        matches!(self, Self::Draft(_))
    }
    pub fn experts(self) -> usize {
        if self.is_draft() { 128 } else { 384 }
    }
    pub fn topk(self) -> usize {
        if self.is_draft() { 3 } else { 6 }
    }
    pub fn router_split(self) -> usize {
        if self.is_draft() { 4 } else { 8 }
    }
}
