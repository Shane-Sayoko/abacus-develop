#include "fssh_replay.h"

#include "source_lcao/module_operator_lcao/fssh_driver.h"

#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

/// @brief Run deterministic boundary and scale tests through the production rescaling rule.
/// @param summary_path Output JSON path for audit diagnostics.
/// @return Zero when every accepted case conserves total energy within roundoff.
int run_velocity_rescale_audit(const std::string& summary_path) {
    struct AuditCase {
        double kinetic_energy;
        double energy_gap;
    };

    const std::vector<AuditCase> cases = {
        {0.0, 0.0}, {0.0, -1.0e-8},
        {1.0e-16, 0.0}, {1.0e-16, 0.5e-16}, {1.0e-16, -0.5e-16}, {1.0e-16, 1.01e-16},
        {1.0e-12, 0.0}, {1.0e-12, 0.5e-12}, {1.0e-12, -0.5e-12}, {1.0e-12, 1.01e-12},
        {1.0e-8, 0.0}, {1.0e-8, 0.5e-8}, {1.0e-8, -0.5e-8}, {1.0e-8, 1.01e-8},
        {1.0e-4, 0.0}, {1.0e-4, 0.5e-4}, {1.0e-4, -0.5e-4}, {1.0e-4, 1.01e-4},
        {1.0, 0.0}, {1.0, 0.5}, {1.0, -0.5}, {1.0, 1.01},
        {1.0e4, 0.0}, {1.0e4, 0.5e4}, {1.0e4, -0.5e4}, {1.0e4, 1.01e4},
    };

    FsshDriver driver;
    std::size_t accepted_cases = 0;
    std::size_t rejected_cases = 0;
    std::size_t failures = 0;
    double max_abs_residual = 0.0;

    for (const AuditCase& audit_case : cases) {
        const FsshVelocityRescaleResult result =
            driver.evaluate_velocity_rescale(audit_case.kinetic_energy, 0.0, audit_case.energy_gap);
        if (!result.accepted) {
            ++rejected_cases;
            continue;
        }

        ++accepted_cases;
        max_abs_residual = std::max(max_abs_residual, std::abs(result.total_energy_residual));
        const double tolerance = 128.0 * std::numeric_limits<double>::epsilon()
                                 * std::max({1.0, std::abs(audit_case.kinetic_energy),
                                             std::abs(audit_case.energy_gap)});
        if (!std::isfinite(result.scale_factor) || !std::isfinite(result.kinetic_energy_after)
            || std::abs(result.total_energy_residual) > tolerance) {
            ++failures;
        }
    }

    std::ofstream output(summary_path);
    if (!output) {
        throw std::runtime_error("cannot write velocity-rescale audit summary: " + summary_path);
    }
    output << std::setprecision(17)
           << "{\n"
           << "  \"audit_cases\": " << cases.size() << ",\n"
           << "  \"accepted_cases\": " << accepted_cases << ",\n"
           << "  \"rejected_cases\": " << rejected_cases << ",\n"
           << "  \"max_abs_total_energy_residual_hartree\": " << max_abs_residual << ",\n"
           << "  \"failures\": " << failures << ",\n"
           << "  \"passed\": " << (failures == 0 ? "true" : "false") << "\n"
           << "}\n";
    std::cout << std::setprecision(10)
              << "FSSH velocity-rescale audit completed: accepted=" << accepted_cases
              << ", rejected=" << rejected_cases
              << ", max energy residual=" << max_abs_residual
              << ", failures=" << failures << '\n';
    return failures == 0 ? 0 : 1;
}

} // namespace

/// @brief Entry point for the file-driven, fixed-nuclear-path FSSH replay tool.
/// @param argc Number of command-line arguments.
/// @param argv Input stream path, comparison CSV path, and summary JSON path.
/// @return Zero for a successful replay, otherwise a non-zero error status.
int main(int argc, char** argv) {
    if (argc == 3 && std::string(argv[1]) == "--velocity-rescale-audit") {
        try {
            return run_velocity_rescale_audit(argv[2]);
        } catch (const std::exception& error) {
            std::cerr << "FSSH velocity-rescale audit failed: " << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 4) {
        std::cerr << "Usage: fssh_replay <input.fssh_replay> <comparison.csv> <summary.json>\n"
                  << "   or: fssh_replay --velocity-rescale-audit <summary.json>\n";
        return 2;
    }
    try {
        FsshReplay replay;
        const FsshReplaySummary summary = replay.run(argv[1], argv[2], argv[3]);
        std::cout << std::setprecision(10)
                  << "FSSH replay completed: steps=" << summary.steps
                  << ", proposed-state mismatches=" << summary.proposed_state_mismatches
                  << ", max probability error=" << summary.max_hop_probability_error
                  << ", max post-decoherence coefficient error=" << summary.max_post_decoherence_error
                  << ", max damping error=" << summary.max_damping_error << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FSSH replay failed: " << error.what() << '\n';
        return 1;
    }
}
