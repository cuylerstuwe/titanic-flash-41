//! Immutable process controls. Resolve environment policy while constructing
//! owners, never while replaying a decode step or entering a transport scope.
//! These are deliberately not causal/cache state and cannot change at runtime.
#[inline]
fn flag(name:&str)->bool {std::env::var(name).as_deref()==Ok("1")}
#[derive(Clone,Copy)]
pub(crate) struct SessionFlags {
    pub long_history:bool,
    pub history32:bool,
    pub acceptance_temperature_screen:bool,
    pub readout_layer_trace:bool,
}
impl SessionFlags {
    pub(crate) fn from_env()->Self {
        Self{long_history:flag("CANADA_SCRATCH_LONG_HISTORY_GENERATION"),
            history32:flag("CANADA_SCRATCH_HISTORY32_SCREEN")||flag("CANADA_SCRATCH_HISTORY32_GENERATION"),
            acceptance_temperature_screen:flag("CANADA_SCRATCH_ACCEPTANCE_TEMPERATURE_SCREEN"),
            readout_layer_trace:flag("CANADA_SCRATCH_READOUT_LAYER_TRACE")}
    }
}
#[derive(Clone,Copy)]
pub(crate) struct NativeRowFlags {
    tiny_sum:bool,tiny_gather:bool,diagnostic:bool,medium:bool,live:bool,
}
impl NativeRowFlags {
    pub(crate) fn from_env()->Self {
        Self{tiny_sum:flag("CANADA_SCRATCH_TINY_APPEND_PREFIX_SUM"),
            tiny_gather:flag("CANADA_SCRATCH_TINY_APPEND_PREFIX_GATHER"),
            diagnostic:flag("CANADA_SCRATCH_NATIVE_ROW_TRANSPORT_SCREEN"),
            medium:flag("CANADA_SCRATCH_MEDIUM_NATIVE_TRANSPORT"),
            live:flag("CANADA_SCRATCH_LIVE_ROW_TRANSPORT")}
    }
    #[inline]
    pub(crate) fn permits(&self,sum:bool,gather:bool,live:usize,capacity:usize)->(bool,bool) {
        let common=self.diagnostic||(capacity==128&&live<=64&&self.medium)
            ||([6,16].contains(&capacity)&&self.live);
        (!sum||self.tiny_sum||common,!gather||self.tiny_gather||common)
    }
}
