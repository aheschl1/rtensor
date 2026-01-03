
# Adding Scalar Operators

To add scalar operators, a few things need to be done in the common case:

1. Add the definition in mod.rs of scalar
2. Add it to the backend mod.rs with specify_trait_scalar_cabal
3. Add it to the cpu implmentation in cpu.rs using impl_cpu_scalar. Also add an _{op}
4. Add it to cuda in scalar.cu, making a struct and declaring the launcher
5. Define cuda bindings in scalar.h
6. Test
