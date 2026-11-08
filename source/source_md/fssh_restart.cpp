#include "fssh_restart.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits.h>
#include <sstream>
#include <stdexcept>
namespace FsshRestart
{
std::string read_file(const std::string &path)
{
    std::ifstream file(path.c_str(), std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("Cannot read restart dependency: " + path);
    }
    std::ostringstream bytes;
    bytes << file.rdbuf();
    if (file.bad())
    {
        throw std::runtime_error("Incomplete restart dependency: " + path);
    }
    return bytes.str();
}
std::string checksum(const std::string &data)
{
    std::uint64_t value = UINT64_C(14695981039346656037);
    for (unsigned char byte : data)
    {
        value ^= byte;
        value *= UINT64_C(1099511628211);
    }
    std::ostringstream text;
    text << std::hex << value;
    return text.str();
}
std::string absolute_path(const std::string &path)
{
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) == nullptr)
    {
        throw std::runtime_error("Cannot resolve restart file: " + path);
    }
    return std::string(resolved);
}
void write(
    const std::string &path,
    int step,
    double temperature,
    const std::string &signature,
    const std::string &structure,
    const std::string &payload)
{
    const std::string temporary = path + ".tmp";
    std::ofstream output(temporary.c_str(), std::ios::binary);
    output << std::setprecision(17) << step << '\n' << temperature << '\n';
    output << "ABACUS_FSSH_RESTART 1\n"
           << step << ' ' << signature << ' ' << checksum(structure) << ' ' << checksum(payload) << ' '
           << payload.size() << '\n';
    output << payload << "END_ABACUS_FSSH_RESTART\n";
    output.close();
    if (!output || std::rename(temporary.c_str(), path.c_str()) != 0)
    {
        throw std::runtime_error("Cannot publish complete FSSH restart: " + path);
    }
}
std::string read(
    const std::string &data,
    int step,
    const std::string &signature,
    const std::string &structure)
{
    std::istringstream input(data);
    int saved_step = -1;
    double temperature = 0.0;
    std::string marker;
    int version = 0;
    if (!(input >> saved_step >> temperature >> marker >> version) || saved_step != step ||
        !std::isfinite(temperature) || marker != "ABACUS_FSSH_RESTART" || version != 1)
    {
        throw std::runtime_error(
            "Missing or invalid merged FSSH restart; legacy Restart_fssh.chk requires validated migration.");
    }
    int payload_step = -1;
    std::string saved_signature;
    std::string structure_digest;
    std::string payload_digest;
    std::size_t size = 0;
    if (!(input >> payload_step >> saved_signature >> structure_digest >> payload_digest >> size) ||
        payload_step != step || saved_signature != signature || structure_digest != checksum(structure) || size == 0 ||
        size > data.size())
    {
        throw std::runtime_error("FSSH restart step, physical configuration or STRU checksum mismatch.");
    }
    if (input.get() != '\n')
    {
        throw std::runtime_error("Invalid FSSH restart payload separator.");
    }
    std::string payload(size, '\0');
    input.read(&payload[0], static_cast<std::streamsize>(size));
    std::string end;
    if (!input || checksum(payload) != payload_digest || !std::getline(input, end) ||
        end != "END_ABACUS_FSSH_RESTART" || (input >> std::ws && !input.eof()))
    {
        throw std::runtime_error("Truncated, corrupt or trailing FSSH restart data.");
    }
    return payload;
}
} // namespace FsshRestart
