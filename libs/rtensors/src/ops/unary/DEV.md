
# Adding Unary Operators

To add unary operators, a few things need to be done in the common 0 argument case:

1. Add the definition in mod.rs of unary
2. Add it to the backend mod.rs with specify_trait_unary_cabal
3. Add it to the cpu implmentation in cpu.rs using impl_cpu_unary. Also add an _{op}
4. Add it to cuda in unary.cu, making a struct and declaring the launcher
5. Define cuda bindings in scalar.h
6. Test
