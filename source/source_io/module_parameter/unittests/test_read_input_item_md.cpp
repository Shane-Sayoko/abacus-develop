#include "source_io/module_parameter/read_input.h"

#include <algorithm>
#include <gtest/gtest.h>
#include <stdexcept>

namespace
{

ModuleIO::Input_Item input_item(const ModuleIO::ReadInput &reader, const std::string &name)
{
    for (const auto &item : reader.get_input_lists())
    {
        if (item.first == name)
        {
            return ModuleIO::Input_Item(item.second);
        }
    }
    throw std::runtime_error("Missing INPUT item: " + name);
}

void read_item(
    const ModuleIO::ReadInput &reader,
    Parameter &parameter,
    const std::string &name,
    const std::string &value)
{
    ModuleIO::Input_Item item = input_item(reader, name);
    item.str_values = {value};
    item.read_value(item, parameter);
}

} // namespace

TEST(FsshInputStateSpace, DefaultAndBooleanMetadata)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    EXPECT_TRUE(parameter.mdp.fssh_ground_state_coupling);
    const ModuleIO::Input_Item item = input_item(reader, "fssh_ground_state_coupling");
    EXPECT_EQ(item.type, "Boolean");
    EXPECT_EQ(item.default_value, "True");
    EXPECT_EQ(item.get_availability(), "md_type==fssh");
    read_item(
        reader,
        parameter,
        "fssh_ground_state_coupling",
        "false");
    EXPECT_FALSE(parameter.mdp.fssh_ground_state_coupling);
    read_item(
        reader,
        parameter,
        "fssh_ground_state_coupling",
        "true");
    EXPECT_TRUE(parameter.mdp.fssh_ground_state_coupling);
}

TEST(FsshInputStateSpace, RejectsExcludedGroundInitialSurface)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    read_item(
        reader,
        parameter,
        "md_type",
        "fssh");
    read_item(
        reader,
        parameter,
        "fssh_ground_state_coupling",
        "false");
    const ModuleIO::Input_Item item = input_item(reader, "fssh_ground_state_coupling");
    EXPECT_EXIT(item.check_value(item, parameter), ::testing::ExitedWithCode(1), "");
    read_item(
        reader,
        parameter,
        "fssh_init_state",
        "1");
    EXPECT_NO_THROW(item.check_value(item, parameter));
    EXPECT_EQ(parameter.mdp.fssh_nstate, 2);
}

TEST(FsshInputStateSpace, RejectsInvalidBoolean)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    EXPECT_EXIT(
        read_item(
            reader,
            parameter,
            "fssh_ground_state_coupling",
            "invalid"),
        ::testing::ExitedWithCode(1), "");
}

TEST(FsshInputStateSpace, SurfaceParameterDefaultsAndDomains)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    EXPECT_EQ(parameter.mdp.decoherence, 0);
    const ModuleIO::Input_Item type = input_item(reader, "md_type");
    const std::vector<std::string> &choices = type.domain.choices();
    EXPECT_NE(std::find(choices.begin(), choices.end(), "fssh"), choices.end());
    const ModuleIO::Input_Item states = input_item(reader, "fssh_nstate");
    EXPECT_FALSE(states.domain.validate_explicit("fssh_nstate", {"1"}).empty());
    EXPECT_TRUE(states.domain.validate_explicit("fssh_nstate", {"2"}).empty());
    const ModuleIO::Input_Item threshold = input_item(reader, "fssh_degen_energy_threshold");
    EXPECT_FALSE(threshold.domain.validate_explicit("fssh_degen_energy_threshold", {"-1"}).empty());
}

TEST(FsshInputStateSpace, RejectsRequestedPoolsBeforeLcaoResetsKpar)
{
    const ModuleIO::ReadInput reader(0);
    const ModuleIO::Input_Item type = input_item(reader, "md_type");
    ModuleIO::InputValues requested;
    requested["md_type"] = ModuleIO::InputValue(std::string("fssh"));
    requested["kpar"] = ModuleIO::InputValue(2);
    EXPECT_FALSE(type.domain.validate_requirements("md_type", requested).empty());
    requested["kpar"] = ModuleIO::InputValue(1);
    EXPECT_TRUE(type.domain.validate_requirements("md_type", requested).empty());
    requested["md_type"] = ModuleIO::InputValue(std::string("nve"));
    requested["kpar"] = ModuleIO::InputValue(2);
    EXPECT_TRUE(type.domain.validate_requirements("md_type", requested).empty());
}

TEST(FsshInputStateSpace, RejectsUnsupportedElectronicAndForceSettings)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    read_item(
        reader,
        parameter,
        "calculation",
        "md");
    read_item(
        reader,
        parameter,
        "basis_type",
        "lcao");
    read_item(
        reader,
        parameter,
        "esolver_type",
        "ks-lr");
    read_item(
        reader,
        parameter,
        "md_type",
        "fssh");
    read_item(
        reader,
        parameter,
        "gamma_only",
        "true");
    read_item(
        reader,
        parameter,
        "cal_force",
        "true");
    read_item(
        reader,
        parameter,
        "cal_syns",
        "true");
    const ModuleIO::Input_Item type = input_item(reader, "md_type");
    EXPECT_NO_THROW(type.check_value(type, parameter));
    read_item(
        reader,
        parameter,
        "nspin",
        "2");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
    read_item(
        reader,
        parameter,
        "nspin",
        "1");
    read_item(
        reader,
        parameter,
        "kpar",
        "2");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
    read_item(
        reader,
        parameter,
        "kpar",
        "1");
    read_item(
        reader,
        parameter,
        "cal_stress",
        "true");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
    read_item(
        reader,
        parameter,
        "cal_stress",
        "false");
    read_item(
        reader,
        parameter,
        "lr_nstates",
        "1");
    read_item(
        reader,
        parameter,
        "fssh_nstate",
        "3");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
}

TEST(FsshInputStateSpace, TripletRequiresClosedShellAndExcludedGroundState)
{
    const ModuleIO::ReadInput reader(0);
    Parameter parameter;
    const ModuleIO::Input_Item spin = input_item(reader, "fssh_spin_channel");
    EXPECT_EQ(parameter.mdp.fssh_spin_channel, "singlet");
    EXPECT_EQ(spin.default_value, "singlet");
    EXPECT_EQ(spin.get_availability(), "md_type==fssh");
    EXPECT_FALSE(spin.domain.validate_explicit("fssh_spin_channel", {"mixed"}).empty());

    read_item(reader, parameter, "calculation", "md");
    read_item(reader, parameter, "basis_type", "lcao");
    read_item(reader, parameter, "esolver_type", "ks-lr");
    read_item(reader, parameter, "md_type", "fssh");
    read_item(reader, parameter, "gamma_only", "true");
    read_item(reader, parameter, "cal_force", "true");
    read_item(reader, parameter, "cal_syns", "true");
    read_item(reader, parameter, "nspin", "2");
    read_item(reader, parameter, "fssh_spin_channel", "triplet");
    read_item(reader, parameter, "fssh_nstate", "3");
    read_item(reader, parameter, "lr_nstates", "2");
    read_item(reader, parameter, "fssh_init_state", "1");
    const ModuleIO::Input_Item type = input_item(reader, "md_type");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
    read_item(reader, parameter, "fssh_ground_state_coupling", "false");
    EXPECT_NO_THROW(type.check_value(type, parameter));
    read_item(reader, parameter, "fssh_nstate", "2");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
    read_item(reader, parameter, "fssh_nstate", "3");
    read_item(reader, parameter, "lr_unrestricted", "true");
    EXPECT_EXIT(type.check_value(type, parameter), ::testing::ExitedWithCode(1), "");
}
