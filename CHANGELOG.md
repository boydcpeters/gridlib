<!-- markdownlint-disable MD024 -->
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Docstrings for multiple functions.
- Updated the README.md file.
- Documentation for readthedocs.
- Function to find the peaks in a GRID spectrum.
- An extra example file

### Changed

- Removed the function `grid_resampling` and made a new more general function called `resample_and_fit` which is also able to perform multi-exponential fitting.
- Added vectorization to the simulation of the survival distributions, which speeds
up the simulation with a factor of ~10-100x.
- Example files.

### Fixed

- Fixed a lot of docstrings where the structure was incorrectly given.
- Docstrings of the functions in the simulate.py file.

## [0.4.1] - 2022-12-03

### Fixed

- Fixed markdownlint errors in CHANGELOG.md file.
- Fixed bug in the event spectrum plotting.

## [0.4.0] - 2022-12-03

### Added

- Added dependencies: psutil and tqdm. These are required for the resampling function.
- Added gridlib default colormap.
- Added function that plots the resampling results as a heatmap for both event and state spectrum.
- Added multiprocessing option to the resampling.
- Functions to read and write GRID resampling results in .mat files.
- Examples to run GRIDLib.

### Fixed

- Fixed a bug in the simulation function, which lead to incorrect delta t's between sample points.
- Set the weight of the decay rate of the single-exponential in the event spectrum to 0.2 to improve visualization.

## [0.3.6] - 2022-11-07

### Fixed

- Bug in the plotting of the survival function curves is now actually fixed, since it was not fixed in the previous version due to inconsistant api, which will be fixed in a later version.

## [0.3.5] - 2022-11-07

### Fixed

- Thought that a bug in the plotting of survival functions was fixed. This is not the case.

## [0.3.4] - 2022-11-06

- Bug was not properly fixed yet, but should be now.

## [0.3.3] - 2022-11-06

- Bug was not properly fixed yet, but should be now.

## [0.3.2] - 2022-11-06

- Fixed bug in a plotting function for plotting the GRID curves against the data.

## [0.3.1] - 2022-11-06

- Fixed a bug which prevented the read_track_file_csv() function from being called.

## [0.3.0] - 2022-11-06

- Added io functions for track data reading and writing, this is an initial version and will change later to make it more general.
- Added function to retrieve the track lifes of a track.
- Added myst-parser to dev dependencies for the docs.
- Removed recommonmark from dev dependencies, since it is deprecated.

## [0.2.1] - 2022-11-05

- Performed some bug fixes.

## [0.2.0] - 2022-11-04

- Added the reference to the original paper.

## [0.1.0] - 2022-08-05

### Added

- First release on PyPI.

[Unreleased]: https://github.com/boydcpeters/gridlib/compare/v0.4.1...HEAD
[0.4.1]: https://github.com/boydcpeters/gridlib/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/boydcpeters/gridlib/compare/v0.3.6...v0.4.0
[0.3.6]: https://github.com/boydcpeters/gridlib/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/boydcpeters/gridlib/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/boydcpeters/gridlib/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/boydcpeters/gridlib/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/boydcpeters/gridlib/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/boydcpeters/gridlib/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/boydcpeters/gridlib/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/boydcpeters/gridlib/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/boydcpeters/gridlib/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/boydcpeters/gridlib/compare/releases/tag/v0.1.0
