// Notebook-only configuration for xeus-cling / cling.
// This header exists purely to keep notebooks tidy.
//
// NOTE: This header is intentionally "re-include friendly".
// In notebooks, different cells may want to enable optional features
// (Postgres / Matplot++) after other cells already included this file.
// Using `#pragma once` would prevent those later feature blocks from running.

// Usage in a notebook cell:
//   #define MCPPFA_NOTEBOOK_ENABLE_PG 1       // optional
//   #define MCPPFA_NOTEBOOK_ENABLE_MATPLOT 1  // optional
//   #include "include/mcppfa/notebook_setup.hpp"

// --- Base setup (run once) ---
#ifndef MCPPFA_NOTEBOOK_SETUP_BASE_INCLUDED
#define MCPPFA_NOTEBOOK_SETUP_BASE_INCLUDED 1

// Local project headers
#pragma cling add_include_path("include")

// Vendored DataFrame headers
#pragma cling add_include_path("vendor/DataFrame-2.0.0/include")

// Matplot++ headers (vendored)
#pragma cling add_include_path("vendor/matplotplusplus/source")

// --- Common includes used across notebooks ---
#include "mcppfa/mcppfa.hpp"

// Plot helpers (repo-local)
#include "plotpp/plotpp.hpp"

// DataFrame headers (vendored)
#include <DataFrame/DataFrame.h>
#include <DataFrame/DataFrameStatsVisitors.h>

#endif // MCPPFA_NOTEBOOK_SETUP_BASE_INCLUDED

// --- Optional: Postgres client libs (libpq / libpqxx) ---
// This is only needed if you use pqxx-backed helpers (e.g. mcppfa::select_to_dataframe with a PG connection string).
// We keep it opt-in so notebooks that don't use Postgres won't fail on systems without these libs.
#if defined(MCPPFA_NOTEBOOK_ENABLE_PG) && !defined(MCPPFA_NOTEBOOK_SETUP_PG_INCLUDED)
#define MCPPFA_NOTEBOOK_SETUP_PG_INCLUDED 1

	// If notebook code requests Postgres helpers, enable the pqxx-backed layer.
	#if !defined(MCPPFA_ENABLE_PG)
		#define MCPPFA_ENABLE_PG 1
	#endif

	#if __has_include(<pqxx/pqxx>)
		// Prefer conda env libs when CONDA_PREFIX is set; keep common system fallback locations.
		#pragma cling add_library_path("$CONDA_PREFIX/lib")
		#pragma cling add_library_path("/usr/lib/x86_64-linux-gnu")
		#pragma cling add_library_path("/usr/local/lib")

		// Load by SONAME (lets the dynamic loader pick the right location)
		#pragma cling load("libpq.so")
		#pragma cling load("libpqxx.so")

		// Ensure pqxx-backed helpers are available even if the base header was already included.
		#include "mcppfa/psql_dataframe.hpp"
	#else
		#warning "MCPPFA_NOTEBOOK_ENABLE_PG set, but <pqxx/pqxx> headers were not found. Install libpqxx (e.g. conda-forge libpqxx) or disable PG."
	#endif
#endif

// --- Optional: Matplot++ shared library ---
// Matplot++ can be used header-only for some parts, but the notebook examples in this repo expect a shared lib build.
// We keep it opt-in so notebooks won't error if Matplot++ hasn't been built yet.
#if defined(MCPPFA_NOTEBOOK_ENABLE_MATPLOT) && !defined(MCPPFA_NOTEBOOK_SETUP_MATPLOT_INCLUDED)
#define MCPPFA_NOTEBOOK_SETUP_MATPLOT_INCLUDED 1
	#include <matplot/matplot.h>
	#pragma cling add_library_path("vendor/matplotplusplus/build-xcpp17/source/matplot")
	#pragma cling load("vendor/matplotplusplus/build-xcpp17/source/matplot/libmatplot.so")
#endif
