#ifndef ABACUS_MD_FSSH_RESTART_H
#define ABACUS_MD_FSSH_RESTART_H
#include <string>
namespace FsshRestart
{
/** @brief Read a complete file. @param[in] path File path. @return Exact bytes, or throw on I/O failure. */
std::string read_file(const std::string &path);
/** @brief Compute an integrity checksum. @param[in] data Bytes. @return Hexadecimal FNV-1a checksum. */
std::string checksum(const std::string &data);
/** @brief Resolve a file for diagnostics. @param[in] path Existing file. @return Absolute path, or throw. */
std::string absolute_path(const std::string &path);
/**
 * @brief Publish a complete FSSH state in the official Restart_md.txt file using atomic rename.
 * @param[in] path Official restart path; temporary sibling is removed by rename.
 * @param[in] step Completed MD step, zero based.
 * @param[in] temperature Nuclear initial temperature in Hartree.
 * @param[in] signature Physical configuration checksum.
 * @param[in] structure Exact STRU_MD bytes for the same step.
 * @param[in] payload Electronic state in atomic units, including its gauge and RNG state.
 */
void write(
    const std::string &path,
    int step,
    double temperature,
    const std::string &signature,
    const std::string &structure,
    const std::string &payload);
/**
 * @brief Validate a complete restart before returning its electronic state.
 * @param[in] data Official Restart_md.txt bytes.
 * @param[in] step Expected completed step.
 * @param[in] signature Current physical configuration checksum.
 * @param[in] structure Exact corresponding STRU_MD bytes.
 * @return Validated electronic payload; malformed, mismatched or legacy data throw.
 */
std::string read(
    const std::string &data,
    int step,
    const std::string &signature,
    const std::string &structure);
} // namespace FsshRestart
#endif
