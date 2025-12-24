//! Procedural macro for generating RPC client routines from an impl block.
//!
//! This crate provides the `#[routines]` attribute macro, which, given an impl block and an enum type,
//! generates RPC client routines for each method in the impl block. Methods can be customized with the
//! `#[rpc(...)]` attribute to control code generation.

use std::cell::RefCell;
use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{parenthesized, parse::{Parse, ParseStream}, parse_macro_input, parse_quote, Expr, Ident, ItemImpl, PathArguments, Token, Type};
use syn::Result;


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
    let mut impl_block = parse_macro_input!(item as ItemImpl);
    let rpc_args = parse_macro_input!(attr as RpcArgs);
    let enum_variant = rpc_args.target_enum;

    for item in &mut impl_block.items {
        if let syn::ImplItem::Fn(method) = item {
            // Parse #[rpc(...)] attribute if present
            let rpc_args = if let Some(attr) = rpc_attr(&method.attrs) {
                let args = RpcMethodArgs::parse(attr).expect("Failed to parse args");
                // Remove the helper attribute so it doesn't appear in output
                method.attrs.retain(|a| !a.path().is_ident("rpc"));
                Some(args)
            } else {
                None
            }.unwrap_or_default();
            if rpc_args.skip {
                continue;
            }
            let (message_signature, response_signature) = process_method(method, &rpc_args);

            let mut new_method = method.clone();
            let variant_name = rpc_args.override_variant_name.as_ref().unwrap_or(&message_signature.name);
            let initializers = &message_signature.initializers;
            let response_name = &response_signature.name;
            // Output conversion logic based on return type
            let out_conv = match &new_method.sig.output {
                syn::ReturnType::Type(_, ty) => {
                    if let syn::Type::Path(type_path) = &**ty {
                        if !type_is_result(type_path) {
                            syn::Error::new_spanned(ty, "RPC method must return a Result type. Consider skipping and implementing the routine yourself.").to_compile_error()
                        } else if type_is_unit_result(type_path) {
                            quote! { result.into() }
                        } else {
                            quote! { result?.into() }
                        }
                    } else {
                        quote! { result.into() }
                    }
                },
                _ => quote! { () },
            };
            let sync_tokens = if rpc_args.sync {
                quote! {
                    self.sync();
                }
            } else {
                quote! {}
            };
            new_method.block = parse_quote!({
                #sync_tokens
                let message = #enum_variant::#variant_name {
                    #( #initializers ),*
                };

                let receiver = self.send_message(message);
                let response = receiver.recv()
                    .map_err(|e| TensorError::BackendError(format!("Failed to receive RPC response: {}", e)))?;
                match response {
                    #enum_variant::#response_name(result) => {
                        #out_conv
                    },
                    _ => {
                        return Err(TensorError::BackendError("Received unexpected RPC response message".to_string()));
                    }
                }
            });

            *item = syn::ImplItem::Fn(new_method);
        }
    }
    let output = quote! {
        #impl_block
    };
    output.into()
}


/// Holds the name and field initializers for an RPC message or response variant.
struct MessageSignature {
    /// The enum variant name.
    name: Ident,
    /// Field initializers for the variant.
    initializers: Vec<proc_macro2::TokenStream>,
}


/// Processes a method and its RPC arguments, returning the message and response variant signatures.
fn process_method(
    method: &mut syn::ImplItemFn,
    rpc_args: &RpcMethodArgs,
) -> (MessageSignature, MessageSignature) {
    let method_name = &method.sig.ident;
    let variant_name = format_ident!("{}", from_snake_to_camel(&method_name.to_string()));
    let response_variant_name = format_ident!("{}Response", variant_name);

    let initializers = RefCell::new(Vec::new());
    for arg in method.sig.inputs.iter() {
        if let syn::FnArg::Typed(pat) = arg {
            let name = &pat.pat;
            let ty = &pat.ty;
            let init = if let syn::Type::Tuple(tuple_type) = &**ty {
                // For tuple arguments, convert each element with .into()
                let inits = tuple_type.elems.iter().enumerate().map(|(i, _)| {
                    let index = syn::Index::from(i);
                    quote! { #name.#index.into() }
                });
                quote! { ( #( #inits ),* ) }
            } else {
                quote! { #name.into() }
            };
            initializers.borrow_mut().push(quote! { #name: #init });
        }
    }
    for arg in rpc_args.extra_variant_args.iter() {
        let name = &arg.name;
        let init_arg = &arg.extract_expr;
        initializers.borrow_mut().push(quote! { #name: #init_arg });
    }
    let signature = MessageSignature {
        name: variant_name.clone(),
        initializers: initializers.borrow().clone(),
    };
    let response_signature = MessageSignature {
        name: response_variant_name.clone(),
        initializers: vec![],
    };
    (signature, response_signature)
}


/// Arguments to the `#[routines]` macro, specifying the target enum.
#[derive(Debug)]
struct RpcArgs {
    /// The enum type to use for messages and responses.
    target_enum: Ident,
}

impl Parse for RpcArgs {
    fn parse(input: ParseStream) -> Result<Self> {
        let ident: Result<Ident> = input.parse();
        Ok(RpcArgs { target_enum: ident? })
    }
}


/// Converts a snake_case string to CamelCase.
#[inline]
fn from_snake_to_camel(s: &str) -> String {
    s.split('_')
        .map(|word| {
            let mut c = word.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        })
        .collect()
}



/// Represents an extra argument to be injected into the generated RPC method.
#[derive(Debug)]
struct ExtraVariantArg {
    /// The argument name.
    name: Ident,
    /// The expression to extract the value.
    extract_expr: Expr,
}

impl Parse for ExtraVariantArg {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let name: Ident = input.parse()?;
        input.parse::<Token![:]>()?;
        let _ty: Type = input.parse()?;
        input.parse::<Token![=]>()?;
        let extract_expr: Expr = input.parse()?;
        Ok(ExtraVariantArg { name, extract_expr })
    }
}


/// Arguments parsed from the `#[rpc(...)]` attribute on a method.
#[derive(Default)]
struct RpcMethodArgs {
    /// If true, skip code generation for this method.
    skip: bool,
    /// Extra arguments to inject into the message.
    extra_variant_args: Vec<ExtraVariantArg>,
    /// enum variant name
    override_variant_name: Option<Ident>,
    // sync before action'
    sync: bool,
}


impl RpcMethodArgs {
    /// Parses the `#[rpc(...)]` attribute.
    fn parse(attr: &syn::Attribute) -> syn::Result<Self> {
        let mut args = RpcMethodArgs::default();
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                args.skip = true;
                Ok(())
            } else if meta.path.is_ident("extra") {
                let content;
                parenthesized!(content in meta.input);
                let parsed = content.parse_terminated(ExtraVariantArg::parse, Token![,])?;
                args.extra_variant_args.extend(parsed);
                Ok(())
            } else if meta.path.is_ident("variant") {
                let content;
                parenthesized!(content in meta.input);
                let ident: Ident = content.parse()?;
                args.override_variant_name = Some(ident);
                Ok(())
            } else if meta.path.is_ident("sync") {
                args.sync = true;
                Ok(())
            } else {
                Err(meta.error("unsupported rpc attribute argument"))
            }
        })?;
        Ok(args)
    }
}


/// Finds the `#[rpc(...)]` attribute in a list of attributes, if present.
fn rpc_attr(attrs: &[syn::Attribute]) -> Option<&syn::Attribute> {
    attrs.iter().find(|a| a.path().is_ident("rpc"))
}


/// Returns true if the type is `Result<(), _>`.
fn type_is_unit_result(ty: &syn::TypePath) -> bool {
    let segment = match ty.path.segments.last() {
        Some(seg) => seg,
        None => return false,
    };
    if segment.ident != "Result" {
        return false;
    }
    let args = match &segment.arguments {
        PathArguments::AngleBracketed(args) => &args.args,
        _ => return false,
    };
    if args.len() != 2 {
        return false;
    }
    match args.first().unwrap() {
        syn::GenericArgument::Type(Type::Tuple(tup)) if tup.elems.is_empty() => true,
        _ => false,
    }
}


/// Returns true if the type is a `Result<_, _>`.
fn type_is_result(ty: &syn::TypePath) -> bool {
    use syn::PathArguments;
    let segment = match ty.path.segments.last() {
        Some(seg) => seg,
        None => return false,
    };
    if segment.ident != "Result" {
        return false;
    }
    match &segment.arguments {
        PathArguments::AngleBracketed(args) => args.args.len() == 2,
        _ => false,
    }
}
