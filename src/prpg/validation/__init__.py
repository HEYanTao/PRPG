"""Independent validation modules.

The package initializer deliberately performs no eager submodule imports.
Validation modules sit on both sides of the simulation/storage boundary, so
package-level re-exports create circular imports during CLI startup.  Import
the concrete module that owns a contract instead.
"""
