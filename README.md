[![Rustfmt](https://github.com/jordan-schnur/SandGears/actions/workflows/.rustfmt.yml/badge.svg?branch=main)](https://github.com/jordan-schnur/SandGears/actions/workflows/.rustfmt.yml)
# SandGears
### Inspired By [Powder Toy](https://powdertoy.co.uk/)

SandGears is an interactive 2D sandbox simulation that allows users to experiment with mechanical contraptions and granular materials like sand. Play around with pistons, hinges, heaters, and more to create unique contraptions and observe their effects on the sand environment.
<img alt="intro gif" align="left" width="100" height="200" src="https://raw.githubusercontent.com/jordan-schnur/SandGears/main/intro.gif">

## Getting Started

To get started with SandGears, clone the repository and follow the instructions for building and running the project on your platform.

bash

```sh
cargo run
cd SandGears
```

### Prerequisites

-   [Vulkan SDK](https://www.lunarg.com/vulkan-sdk/)
    - [Vulkanalia](https://github.com/KyleMayes/vulkanalia) has a great [tutorial](https://kylemayes.github.io/vulkanalia/development_environment.html) on how to get started. https://kylemayes.github.io/vulkanalia/development_environment.html

### Building and Running

```sh
cargo run
```

## Contributing

Please check the [Issues](https://github.com/jordan-schnur/SandGears/issues) section for more specific tasks and feature requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## TODOs

- [x]  Implement basic sand simulation with a 2D grid.
- [x]  Add more particles...
   - [x] Water
   - [ ] Fire
   - [ ] Lava
   - [x] Gas
   - [ ] Snow
   - [x] Sand
- [ ]  Implement text rendering
- [ ]  Add support for mechanical contraptions (pistons, hinges, etc.).
- [ ]  Implement a heater that can melt sand.
- [ ]  Create a user-friendly interface for adding and interacting with contraptions.
- [ ]  Develop a system for saving and loading user-created scenes.
- [ ]  Optimize the simulation for better performance.
- [ ]  Add more types of granular materials with different properties.
- [ ]  Cross-platform builds
- [ ]  Sign builds from [here](https://shop.certum.eu/open-source-code-signing-code.html)
- [ ]  Test on various platforms and fix platform-specific issues.
