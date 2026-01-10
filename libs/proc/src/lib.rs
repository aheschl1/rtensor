use proc_macro::TokenStream;

mod grad;
mod rpc;

#[proc_macro_attribute]
/// Attribute macro to mark a routine as requiring an active gradient context.
/// If no gradient context is active at runtime, an error will be raised.
/// 
/// # Usage
/// 
/// ```ignore
/// #[requires_grad]
/// fn my_function(...) -> ... {
///     // ...
/// }
/// ```
/// 
/// Expands to
/// ```ignore
/// fn my_function(...) -> ... {
///    grad::when_enabled::<T, B, _>(|ctx| {
///        // original function body
///    }).expect("Gradient context required but not found.")
/// }
/// ```
/// 
/// When applied to an `impl` block, all methods within the block are wrapped similarly.
pub fn when_enabled(attr: TokenStream, item: TokenStream) -> TokenStream {
    grad::when_enabled(attr, item)
}

#[proc_macro_attribute]
/// Attribute macro to generate RPC client routines for each method in an impl block.
///
/// # Usage
///
/// ```ignore
/// #[routines(MyRpcEnum)]
/// impl MyClient {
///     // ...
/// }
/// ```
///
/// Optionally, methods can be annotated with `#[rpc(skip)]` to skip codegen, or `#[rpc(extra(...))]` to add extra arguments.
/// By default, the variant for the method is derived from the method name in CamelCase.
/// To override the variant name, use `#[rpc(variant(VariantName))]`.
pub fn routines(attr: TokenStream, item: TokenStream) -> TokenStream {
    rpc::routines(attr, item)
}
