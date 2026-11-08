#include "source_md/fssh_restart.h"
#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

TEST(FsshRestart, OfficialHeaderAndAtomicRoundTrip)
{
    const std::string path = testing::TempDir() + "Restart_md_fssh_test.txt";
    FsshRestart::write(
        path,
        3,
        0.001,
        "model",
        "structure",
        "electrons\n");
    const std::string bytes = FsshRestart::read_file(path);
    EXPECT_EQ(
        FsshRestart::read(
            bytes,
            3,
            "model",
            "structure"),
        "electrons\n");
    EXPECT_THROW(
        FsshRestart::read(
            bytes,
            4,
            "model",
            "structure"),
        std::runtime_error);
    EXPECT_THROW(
        FsshRestart::read(
            bytes,
            3,
            "other",
            "structure"),
        std::runtime_error);
    EXPECT_THROW(
        FsshRestart::read(
            bytes,
            3,
            "model",
            "changed"),
        std::runtime_error);
    EXPECT_THROW(
        FsshRestart::read(
            bytes.substr(0, bytes.size() - 4),
            3,
            "model",
            "structure"),
        std::runtime_error);
    EXPECT_THROW(
        FsshRestart::read(
            bytes + "extra",
            3,
            "model",
            "structure"),
        std::runtime_error);
    std::string damaged = bytes;
    damaged[damaged.find("electrons")] = 'X';
    EXPECT_THROW(
        FsshRestart::read(
            damaged,
            3,
            "model",
            "structure"),
        std::runtime_error);
    EXPECT_THROW(
        FsshRestart::read(
            "3\n0.001\n",
            3,
            "model",
            "structure"),
        std::runtime_error);
    // A failed temporary write must leave the last committed snapshot readable.
    std::ofstream interrupted((path + ".tmp").c_str());
    interrupted << "partial";
    interrupted.close();
    EXPECT_EQ(FsshRestart::read_file(path), bytes);
    FsshRestart::write(
        path,
        4,
        0.001,
        "model",
        "new structure",
        "new electrons\n");
    EXPECT_EQ(
        FsshRestart::read(
            FsshRestart::read_file(path),
            4,
            "model",
            "new structure"),
        "new electrons\n");
    EXPECT_EQ(std::remove(path.c_str()), 0);
}

TEST(FsshRestart, FailedWritePreservesCommittedState)
{
    const std::string path = testing::TempDir() + "Restart_md_fssh_failure.txt";
    FsshRestart::write(
        path,
        2,
        0.001,
        "model",
        "structure",
        "state\n");
    const std::string original = FsshRestart::read_file(path);
    const std::string temporary = path + ".tmp";
    std::ofstream blocked(temporary.c_str());
    blocked << "interrupted temporary file";
    blocked.close();
    ASSERT_EQ(chmod(temporary.c_str(), 0444), 0);
    EXPECT_THROW(
        FsshRestart::write(
            path,
            3,
            0.001,
            "model",
            "structure2",
            "state2\n"),
        std::runtime_error);
    EXPECT_EQ(FsshRestart::read_file(path), original);
    ASSERT_EQ(chmod(temporary.c_str(), 0600), 0);
    EXPECT_EQ(std::remove(temporary.c_str()), 0);
    EXPECT_EQ(std::remove(path.c_str()), 0);
}
