# Attractor Renderer

This is a program for finding strange attractors, as first described by Paul
Bourke in [this article](http://paulbourke.net/fractals/lyapunov/), and
rendering them using a color gradient, intended to be used as wallpapers.

# Building

## Attractor-Egui

The main crate is `attractor-egui`, which is a Windows, Linux and Mac OS (MacOs
is untested) application. To build it, you need to have a recent version of the
Rust toolchain installed.

```sh
cargo build -p attractor-egui --release
```

Or to run it directly:

```sh
cargo run -p attractor-egui --release
```

## Android

There is also an Android version. To build it, you can to open the `android`
folder in Android Studio, and build it from there. You need to have installed
the Android NDK and SDK (see [build.gradle.kts](android/app/build.gradle.kts)
for the versions used).

# Acknowledgements

Inspired by:

- Finding strange attractors: [Part 1](https://youtu.be/AzdpM-vfUCQ), [Part 2](https://youtu.be/sGdFR9cpE6A)
- Finding Strange Attractors: https://www.youtube.com/watch?v=Lw_SqFxHtH0
- Random Attractors: https://paulbourke.net/fractals/lyapunov/

Some of the code is derived from:

- [ok-picker](https://github.com/gagbo/ok-picker/tree/6b36785955b4318a3e75860e56a755ecca7c3967) ([GPL-3.0-only](https://github.com/gagbo/ok-picker/blob/6b36785955b4318a3e75860e56a755ecca7c3967/LICENSES/GPL-3.0-only.txt))
- [http://jbrd.github.io/2020/12/27/monotone-cubic-interpolation.html](http://jbrd.github.io/2020/12/27/monotone-cubic-interpolation.html)
- [http://bottosson.github.io/misc/ok_color.h](http://bottosson.github.io/misc/ok_color.h) ([MIT](https://bottosson.github.io/posts/colorpicker/#license))
