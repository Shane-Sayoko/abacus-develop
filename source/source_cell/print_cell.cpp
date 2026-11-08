#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unistd.h>

#include "print_cell.h"
#include "source_base/formatter.h"
#include "source_base/global_variable.h"
#include "source_base/output.h"
#include "source_base/parallel_cell.h"
#include "source_base/tool_quit.h"
#include "source_base/tool_title.h"
#include "source_cell/mdcell.h"

#ifdef __MPI
#include <mpi.h>
#endif

namespace unitcell
{
    void print_tau(Atom* atoms,
                   const std::string& Coordinate,
                   const int ntype,
                   const double lat0,
                   std::ofstream &ofs)
    {
        ModuleBase::TITLE("UnitCell", "print_tau");
        // assert (direct || Coordinate == "Cartesian" || Coordinate == "Cartesian_angstrom"); // this line causes abort in unittest ReadAtomPositionsCACXY.
        // previously there are two if-statements, the first is `if(Coordinate == "Direct")` and the second is `if(Coordinate == "Cartesian" || Coordiante == "Cartesian_angstrom")`
        // however the Coordinate can also be value among Cartesian_angstrom_center_xy, Cartesian_angstrom_center_xz, Cartesian_angstrom_center_yz and Cartesian_angstrom_center_xyz

        // if Coordinate has value one of them, this print_tau will not print anything.
        std::regex pattern("Direct|Cartesian(_angstrom)?(_center_(xy|xz|yz|xyz))?");
        assert(std::regex_search(Coordinate, pattern));
        bool direct = (Coordinate == "Direct");

        //----------------------
        // print atom positions
        //----------------------
        std::string table;
        table += direct? " DIRECT COORDINATES\n": FmtCore::format(" CARTESIAN COORDINATES ( UNIT = %15.8f Bohr )\n", lat0);
        table += FmtCore::format("%5s%19s%19s%19s%8s\n", "atom", "x", "y", "z", "mag");
        for(int it = 0; it < ntype; it++)
        {
            for (int ia = 0; ia < atoms[it].na; ia++)
            {
                const double& x = direct? atoms[it].taud[ia].x: atoms[it].tau[ia].x;
                const double& y = direct? atoms[it].taud[ia].y: atoms[it].tau[ia].y;
                const double& z = direct? atoms[it].taud[ia].z: atoms[it].tau[ia].z;
                table += FmtCore::format("%5s%19.12f%19.12f%19.12f%8.4f\n", 
                                        atoms[it].label, 
                                        x, 
                                        y, 
                                        z, 
                                        atoms[it].mag[ia]); 
            }
        }
        table += "\n";
        ofs << table; 


        // print velocities
        ofs << " ATOMIC VELOCITIES" << std::endl;
        ofs << std::setprecision(12);
        ofs << std::setw(5) << "atom" 
            << std::setw(19) << "vx" 
            << std::setw(19) << "vy" 
            << std::setw(19) << "vz"
            << std::endl;
 
        for(int it = 0; it < ntype; it++)
        {
            for (int ia = 0; ia < atoms[it].na; ia++)
            {
                ofs << std::setw(5) << atoms[it].label;
                ofs << " " << std::setw(18) << atoms[it].vel[ia].x;
                ofs << " " << std::setw(18) << atoms[it].vel[ia].y;
                ofs << " " << std::setw(18) << atoms[it].vel[ia].z;
                ofs << std::endl;
            }
        }
        ofs << std::endl;
        ofs << std::setprecision(6); // return to 6, as original


        return;
    }

    void print_stru_file(const UnitCell& ucell,
                         const Atom*     atoms,
                         const ModuleBase::Matrix3& latvec,
                         const std::string& fn,
                         const std::string& header,
                         const int& nspin,
                         const bool& vel,
                         const bool& magmom,
                         const bool& orb,
                         const bool& dpks_desc,
                         const int& iproc,
                         const ModuleBase::matrix& force,
                         const bool& has_force)
    {
        ModuleBase::TITLE("UnitCell","print_stru_file");
        if (iproc != 0)
        {
            return; // old: if(GlobalV::MY_RANK != 0) return;
        }
        // optional header comments
        std::string str;
        if (!header.empty())
        {
            str = header;
        }
        // ATOMIC_SPECIES
        str += "ATOMIC_SPECIES\n";
        for(int it=0; it<ucell.ntype; it++)
        { 
            str += FmtCore::format("%s %8.4f %s %s\n", 
                                    ucell.atoms[it].label, 
                                    ucell.atoms[it].mass, 
                                    ucell.pseudo_fn[it], 
                                    ucell.pseudo_type[it]); 
        }
        // NUMERICAL_ORBITAL
        if(orb)
        {
            str += "\nNUMERICAL_ORBITAL\n";
            for(int it = 0; it < ucell.ntype; it++) 
            { 
                str += ucell.orbital_fn[it] + "\n"; 
            }
        }
        // NUMERICAL_DESCRIPTOR
        if(dpks_desc) 
        { 
            str += "\nNUMERICAL_DESCRIPTOR\n" + ucell.descriptor_file + "\n"; 
        }
        // LATTICE_CONSTANT: fixed to one Angstrom expressed in Bohr, so that the
        // lattice vectors below can be written directly in Angstrom.
        const double lat0_angstrom = 1.0 / ModuleBase::BOHR_TO_A;
        str += "\nLATTICE_CONSTANT\n"
             + FmtCore::format("%-.10f", lat0_angstrom)
             + " # in Bohr (= 1 Angstrom); lattice vectors below are in Angstrom\n";
        // LATTICE_VECTORS: internal vectors are dimensionless multiples of ucell.lat0;
        // multiply by ucell.lat0 * BOHR_TO_A to get the physical vectors in Angstrom.
        const double lat_scale = ucell.lat0 * ModuleBase::BOHR_TO_A;
        str += "\nLATTICE_VECTORS # in Angstrom\n";
        str += FmtCore::format("%.16f %.16f %.16f\n",
                               latvec.e11 * lat_scale, latvec.e12 * lat_scale, latvec.e13 * lat_scale);
        str += FmtCore::format("%.16f %.16f %.16f\n",
                               latvec.e21 * lat_scale, latvec.e22 * lat_scale, latvec.e23 * lat_scale);
        str += FmtCore::format("%.16f %.16f %.16f\n",
                               latvec.e31 * lat_scale, latvec.e32 * lat_scale, latvec.e33 * lat_scale);
        // ATOMIC_POSITIONS
        str += "\nATOMIC_POSITIONS\n";
        int nat_ = 0; // counter iat, for printing out Mulliken magmom who is indexed by iat
        // Force output is controlled by the explicit has_force flag rather than
        // inferred from matrix dimensions, so a zero-allocated matrix is not
        // mistaken for computed forces.
        if (has_force && (force.nr != ucell.nat || force.nc != 3))
        {
            ModuleBase::WARNING_QUIT("print_stru_file",
                                     "has_force is true but force matrix dimensions do not match nat x 3");
        }
        // Positions are always written as Cartesian_angstrom so that the
        // coordinate format is independent of force availability; has_force
        // only controls the per-atom f fields and the unit annotation.
        const std::string unit_note = has_force ? " # positions in Angstrom, forces in eV/Angstrom\n"
                                                : " # positions in Angstrom\n";
        str += "Cartesian_angstrom" + unit_note;
        // Internal Cartesian tau is in units of lat0 (Bohr); convert to Angstrom.
        const double pos_conv = ucell.lat0 * ModuleBase::BOHR_TO_A;
        const double force_conv = ModuleBase::Ry_to_eV / ModuleBase::BOHR_TO_A; // Ry/Bohr to eV/Angstrom
        for(int it = 0; it < ucell.ntype; it++)
        {
            str += "\n" + ucell.atoms[it].label + " #label\n";
            // Output real initial magnetism: for nspin=2 use mag[0], for nspin=4 use norm of m_loc_[0]
            double start_mag = ucell.magnet.start_mag[it];
            if (atoms[it].na > 0) {
                if (nspin == 2) {
                    start_mag = atoms[it].mag[0];
                } else if (nspin == 4) {
                    start_mag = std::sqrt(std::pow(atoms[it].m_loc_[0].x, 2)
                                        + std::pow(atoms[it].m_loc_[0].y, 2)
                                        + std::pow(atoms[it].m_loc_[0].z, 2));
                }
            }
            str += FmtCore::format("%.4f #magnetism (default, overridden by per-atom mag below)\n", start_mag);
            str += FmtCore::format("%d #number of atoms\n", atoms[it].na);
            for(int ia = 0; ia < atoms[it].na; ia++)
            {
                // output position
                const double& x = atoms[it].tau[ia].x;
                const double& y = atoms[it].tau[ia].y;
                const double& z = atoms[it].tau[ia].z;
                str += FmtCore::format("%.10f %.10f %.10f", x*pos_conv, y*pos_conv, z*pos_conv);
                str += FmtCore::format(" m%2d%2d%2d", atoms[it].mbl[ia].x, atoms[it].mbl[ia].y, atoms[it].mbl[ia].z);
                if (vel) // output velocity
                {
                    str += FmtCore::format(" v %.10f %.10f %.10f", atoms[it].vel[ia].x, atoms[it].vel[ia].y, atoms[it].vel[ia].z);
                }
                if (has_force) // output force
                {
                    str += FmtCore::format(" f %.6f %.6f %.6f",
                                           force(nat_, 0)*force_conv,
                                           force(nat_, 1)*force_conv,
                                           force(nat_, 2)*force_conv);
                }
                if (nspin == 2) // output magnetic information
                {
                    if (magmom && !ucell.atom_mulliken.empty()) {
                        str += FmtCore::format(" mag %.4f", ucell.atom_mulliken[nat_][1]);
                    } else {
                        str += FmtCore::format(" mag %.4f", atoms[it].mag[ia]);
                    }
                }
                else if (nspin == 4) // output magnetic information
                {
                    if (magmom && !ucell.atom_mulliken.empty()) {
                        str += FmtCore::format(" mag %.4f %.4f %.4f",
                                                ucell.atom_mulliken[nat_][1],
                                                ucell.atom_mulliken[nat_][2],
                                                ucell.atom_mulliken[nat_][3]);
                    } else {
                        str += FmtCore::format(" mag %.4f %.4f %.4f",
                                                atoms[it].m_loc_[ia].x,
                                                atoms[it].m_loc_[ia].y,
                                                atoms[it].m_loc_[ia].z);
                    }
                }
                str += "\n";
                nat_++;
            }
        }
        std::ofstream ofs(fn.c_str());
        ofs << str;
        ofs.close();
        return;
    }

    void print_cell(const UnitCell& ucell, std::ofstream& ofs)
    {
        ModuleBase::GlobalFunc::OUT(ofs, "print_unitcell()");

        ModuleBase::GlobalFunc::OUT(ofs, "latName", ucell.latName);
        ModuleBase::GlobalFunc::OUT(ofs, "ntype", ucell.ntype);
        ModuleBase::GlobalFunc::OUT(ofs, "nat", ucell.nat);
        ModuleBase::GlobalFunc::OUT(ofs, "lat0", ucell.lat0);
        ModuleBase::GlobalFunc::OUT(ofs, "lat0_angstrom", ucell.lat0_angstrom);
        ModuleBase::GlobalFunc::OUT(ofs, "tpiba", ucell.tpiba);
        ModuleBase::GlobalFunc::OUT(ofs, "omega", ucell.omega);

        output::printM3(ofs, "Lattices Vector (R) : ", ucell.latvec);
        output::printM3(ofs, "Supercell lattice vector : ", ucell.latvec_supercell);
        output::printM3(ofs, "Reciprocal lattice Vector (G): ", ucell.G);
        output::printM3(ofs, "GGT : ", ucell.GGT);

        ofs << std::endl;
        return;
    }
}

namespace
{
std::string mdcell_stru_header(const MDCell& cell, const StruMeta& metadata)
{
    std::ostringstream output;
    output << std::fixed << std::setprecision(17);
    output << "ATOMIC_SPECIES\n";
    for (std::size_t it = 0; it < metadata.species.size(); ++it)
    {
        const StruSpecies& species = metadata.species[it];
        output << cell.type_labels()[it] << " " << std::setprecision(17) << cell.type_masses()[it]
               << std::setprecision(17);
        if (!species.pseudo_file.empty()) output << " " << species.pseudo_file;
        if (!species.pseudo_type.empty()) output << " " << species.pseudo_type;
        output << "\n";
    }
    bool has_orbitals = false;
    for (std::size_t it = 0; it < metadata.species.size(); ++it)
        has_orbitals = has_orbitals || !metadata.species[it].orbital_file.empty();
    if (has_orbitals)
    {
        output << "\nNUMERICAL_ORBITAL\n";
        for (std::size_t it = 0; it < metadata.species.size(); ++it)
            output << metadata.species[it].orbital_file << "\n";
    }
    if (!metadata.descriptor_file.empty()) output << "\nNUMERICAL_DESCRIPTOR\n" << metadata.descriptor_file << "\n";
    output << "\nLATTICE_CONSTANT\n" << cell.lat0() << "\n\nLATTICE_VECTORS\n";
    const ModuleBase::Matrix3& lattice = cell.latvec();
    output << lattice.e11 << " " << lattice.e12 << " " << lattice.e13 << "\n";
    output << lattice.e21 << " " << lattice.e22 << " " << lattice.e23 << "\n";
    output << lattice.e31 << " " << lattice.e32 << " " << lattice.e33 << "\n";
    output << "\nATOMIC_POSITIONS\nCartesian\n";
    return output.str();
}

std::string mdcell_type_header(const MDCell& cell, const StruMeta& metadata, const std::size_t it)
{
    const StruSpecies& species = metadata.species[it];
    std::ostringstream output;
    output << "\n" << cell.type_labels()[it] << " #label\n";
    output << std::fixed << std::setprecision(17) << species.start_mag << " #magnetism\n";
    output << cell.type_atom_counts()[it] << " #number of atoms\n";
    return output.str();
}

}

namespace unitcell
{
StruMeta make_stru_meta(const UnitCell& ucell)
{
    StruMeta metadata;
    metadata.species.resize(static_cast<std::size_t>(ucell.ntype));
    for (int it = 0; it < ucell.ntype; ++it)
    {
        StruSpecies& species = metadata.species[static_cast<std::size_t>(it)];
        if (static_cast<std::size_t>(it) < ucell.pseudo_fn.size()) species.pseudo_file = ucell.pseudo_fn[it];
        if (static_cast<std::size_t>(it) < ucell.pseudo_type.size()) species.pseudo_type = ucell.pseudo_type[it];
        if (static_cast<std::size_t>(it) < ucell.orbital_fn.size()) species.orbital_file = ucell.orbital_fn[it];
        if (static_cast<std::size_t>(it) < ucell.magnet.start_mag.size()) species.start_mag = ucell.magnet.start_mag[it];
    }
    metadata.descriptor_file = ucell.descriptor_file;
    return metadata;
}
}

namespace mdcell
{
/**
 * @brief Write restart atoms in stable species/type-index order across MPI decompositions.
 * @param[in] cell Distributed nuclear state; each atom must have exactly one owner.
 * @param[in] stru_meta Species metadata matching the cell.
 * @param[in] fn Destination STRU path. Only rank zero writes, in bounded batches.
 */
void print_stru_file(const MDCell& cell, const StruMeta& stru_meta, const std::string& fn)
{
    if (stru_meta.species.size() != cell.type_labels().size()
        || stru_meta.species.size() != cell.type_masses().size()
        || stru_meta.species.size() != cell.type_atom_counts().size())
    {
        throw std::runtime_error("MDCell STRU metadata does not match the MDCell type data.");
    }
    ModuleBase::CommunicationDomain domain;
#ifdef __MPI
    domain.initialize(cell.communicator());
#endif
    std::ofstream output;
    if (domain.rank() == 0)
    {
        output.open(fn.c_str());
        output << mdcell_stru_header(cell, stru_meta);
        output << std::scientific << std::setprecision(17);
    }
    int invalid = 0;
    const std::int64_t batch_size = 1024;
    for (std::size_t type = 0; type < stru_meta.species.size(); ++type)
    {
        if (domain.rank() == 0)
        {
            output << mdcell_type_header(cell, stru_meta, type);
        }
        const std::int64_t count = cell.type_atom_counts()[type];
        std::vector<const LocalAtom *> local_atoms;
        for (const LocalAtom &atom : cell.owned_atoms())
        {
            if (atom.type == static_cast<int>(type))
            {
                if (atom.type_index < 0 || atom.type_index >= count)
                {
                    invalid = 1;
                }
                else
                {
                    local_atoms.push_back(&atom);
                }
            }
        }
        std::sort(
            local_atoms.begin(), local_atoms.end(),
            [](const LocalAtom *left, const LocalAtom *right) { return left->type_index < right->type_index; });
        std::size_t cursor = 0;
        for (std::int64_t begin = 0; begin < count; begin += batch_size)
        {
            const int size = static_cast<int>(std::min(batch_size, count - begin));
            std::vector<double> records(size * 9, 0.0);
            std::vector<int> owners(size, 0);
            while (cursor < local_atoms.size() && local_atoms[cursor]->type_index < begin + size)
            {
                const LocalAtom &atom = *local_atoms[cursor];
                ++cursor;
                const int index = static_cast<int>(atom.type_index - begin);
                ++owners[index];
                for (int axis = 0; axis < 3; ++axis)
                {
                    records[index * 9 + axis] = atom.cart[axis];
                    records[index * 9 + 3 + axis] = atom.vel[axis];
                    records[index * 9 + 6 + axis] = atom.mbl[axis];
                }
            }
            domain.sum(records.data(), size * 9);
            domain.sum(owners.data(), size);
            for (int index = 0; index < size; ++index)
            {
                if (owners[index] != 1)
                {
                    invalid = 1;
                }
                if (domain.rank() == 0)
                {
                    output << records[index * 9] << ' ' << records[index * 9 + 1] << ' ' << records[index * 9 + 2]
                           << " m " << static_cast<int>(records[index * 9 + 6]) << ' '
                           << static_cast<int>(records[index * 9 + 7]) << ' '
                           << static_cast<int>(records[index * 9 + 8]) << " v " << records[index * 9 + 3] << ' '
                           << records[index * 9 + 4] << ' ' << records[index * 9 + 5] << '\n';
                }
            }
        }
    }
    if (domain.rank() == 0)
    {
        output.close();
        if (!output)
        {
            invalid = 1;
        }
    }
    if (domain.max(invalid) != 0)
    {
        throw std::runtime_error("Incomplete MDCell restart STRU or invalid atom ownership: " + fn);
    }
}
}
