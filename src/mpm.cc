#include <memory>

#include "factory.h"
#include "io.h"

#include "solvers/mpm.h"

// Standard MPM solver
#include "solvers/dry_soil/mpm_explicit.h"
#include "solvers/dry_soil/tm_mpm_explicit.h"
#include "solvers/dry_soil/thermo_mpm_implicit.h"

// Saturated porous media
#include "solvers/saturated_soil/hm_mpm_explicit_twophase.h"
#include "solvers/saturated_soil/hm_mpm_semi_implicit_twophase.h"
#include "solvers/saturated_soil/thm_mpm_explicit_twophase.h"
#include "solvers/saturated_soil/thm_mpm_semi_implicit_twophase.h"

// Unsaturated porous media
#include "solvers/unsaturated_soil/thm_mpm_explicit_threephase.h"

// Frozen porous media
#include "solvers/frozen_soil/th_mpm_explicit_saturated_frozen.h"
#include "solvers/frozen_soil/thm_mpm_explicit_saturated_frozen.h"
#include "solvers/frozen_soil/thm_mpm_semi_implicit_saturated_frozen.h"
#include "solvers/frozen_soil/thm_mpm_semi_implicit_unsaturated_frozen.h"

// Hydrate-bearing sediments
#include "solvers/hydrate_soil/thc_mpm_explicit_hydrate.h"
#include "solvers/hydrate_soil/thmc_mpm_explicit_hydrate.h"

namespace mpm {
// Stress update method
std::map<std::string, StressUpdate> stress_update = {
    {"usf", StressUpdate::USF},
    {"usl", StressUpdate::USL},
    {"musl", StressUpdate::MUSL}};
}  // namespace mpm

// 2D Explicit MPM
static Register<mpm::MPM, mpm::MPMExplicit<2>, const std::shared_ptr<mpm::IO>&>
    mpm_explicit_2d("MPMExplicit2D");

// 3D Explicit MPM
static Register<mpm::MPM, mpm::MPMExplicit<3>, const std::shared_ptr<mpm::IO>&>
    mpm_explicit_3d("MPMExplicit3D");

// 2D Implicit Thermo MPM
static Register<mpm::MPM, mpm::ThermoMPMImplicit<2>, const std::shared_ptr<mpm::IO>&>
    thermo_mpm_implicit_2d("ThermoMPMImplicit2D");

// 3D Implicit Thermo MPM
static Register<mpm::MPM, mpm::ThermoMPMImplicit<3>, const std::shared_ptr<mpm::IO>&>
    thermo_mpm_implicit_3d("ThermoMPMImplicit3D");

// 2D Explicit Thermo-mechnical MPM
static Register<mpm::MPM, mpm::TMMPMExplicit<2>, const std::shared_ptr<mpm::IO>&>
    tm_mpm_explicit_2d("TMMPMExplicit2D");

// 3D Explicit Thermo-mechnical MPM
static Register<mpm::MPM, mpm::TMMPMExplicit<3>, const std::shared_ptr<mpm::IO>&>
    tm_mpm_explicit_3d("TMMPMExplicit3D");

// 2D Explicit Two Phase MPM for hydromechanical modeling
static Register<mpm::MPM, mpm::HMMPMExplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    hm_mpm_explicit_twophase_2d("HMMPMExplicitTwoPhase2D");

// 3D Explicit Two Phase MPM
static Register<mpm::MPM, mpm::HMMPMExplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    hm_mpm_explicit_twophase_3d("HMMPMExplicitTwoPhase3D");

// 2D Explicit Two Phase Thermo-Hydro-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMExplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_twophase_2d("THMMPMExplicitTwoPhase2D");

// 3D Explicit Two Phase Thermo-Hydro-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMExplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_twophase_3d("THMMPMExplicitTwoPhase3D");

// 2D Explicit Three Phase Thermo-Hydro-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMExplicitThreePhase<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_threephase_2d("THMMPMExplicitThreePhase2D");

// 3D Explicit Three Phase Thermo-Hydro-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMExplicitThreePhase<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_threephase_3d("THMMPMExplicitThreePhase3D");

// 2D Semi-Implicit Two Phase Hydro-mechnical MPM
static Register<mpm::MPM, mpm::HMMPMSemiImplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    hm_mpm_semi_implicit_twophase_2d("HMMPMSemiImplicitTwoPhase2D");

// 3D Semi-Implicit Two Phase Hydro-mechnical MPM
static Register<mpm::MPM, mpm::HMMPMSemiImplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    hm_mpm_semi_implicit_twophase_3d("HMMPMSemiImplicitTwoPhase3D");

// 2D SemiImplicit Two Phase Thermo-hydro-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMSemiImplicitTwoPhase<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_twophase_2d("THMMPMSemiImplicitTwoPhase2D");

// 3D Explicit Two Phase Thermo-mechnical MPM
static Register<mpm::MPM, mpm::THMMPMSemiImplicitTwoPhase<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_twophase_3d("THMMPMSemiImplicitTwoPhase3D");

// 2D Explicit THM-MPM with phase change
static Register<mpm::MPM, mpm::THMPMExplicitSatFrozen<2>,
                const std::shared_ptr<mpm::IO>&>
    th_mpm_explicit_saturated_frozen_2d("THMPMExplicitSatFrozen2D");

// 3D SemiImplicit THM-MPM with phase change
static Register<mpm::MPM, mpm::THMPMExplicitSatFrozen<3>,
                const std::shared_ptr<mpm::IO>&>
    th_mpm_explicit_saturated_frozen_3d("THMPMExplicitSatFrozen3D");

// 2D Explicit THM-MPM for saturated frozen soil
static Register<mpm::MPM, mpm::THMMPMExplicitSatFrozen<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_sat_frozen_2d("THMMPMExplicitSatFrozen2D");

// 3D Explicit THM-MPM for saturated frozen soil
static Register<mpm::MPM, mpm::THMMPMExplicitSatFrozen<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_explicit_sat_frozen_3d("THMMPMExplicitSatFrozen3D");

// 2D Semi-Implicit THM-MPM for saturated frozen soil
static Register<mpm::MPM, mpm::THMMPMSemiImplicitSatFrozen<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_sat_frozen_2d("THMMPMSemiImplicitSatFrozen2D");

// 3D Semi-Implicit THM-MPM for saturated frozen soil
static Register<mpm::MPM, mpm::THMMPMSemiImplicitSatFrozen<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_sat_frozen_3d("THMMPMSemiImplicitSatFrozen3D");

// 2D Semi-Implicit THM-MPM for unsaturated frozen soil
static Register<mpm::MPM, mpm::THMMPMSemiImplicitUnsatFrozen<2>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_unsat_frozen_2d("THMMPMSemiImplicitUnsatFrozen2D");

// 3D Semi-Implicit THM-MPM for unsaturated frozen soil
static Register<mpm::MPM, mpm::THMMPMSemiImplicitUnsatFrozen<3>,
                const std::shared_ptr<mpm::IO>&>
    thm_mpm_semi_implicit_unsat_frozen_3d("THMMPMSemiImplicitUnsatFrozen3D");

// 2D Explicit Thermo-Hydro-Chemical MPM for Hydrate-Bearing Sediment
static Register<mpm::MPM, mpm::THCMPMExplicitHydrate<2>,
                const std::shared_ptr<mpm::IO>&>
    thc_mpm_explicit_hydrate_2d("THCMPMExplicitHydrate2D");

// 3D Explicit Thermo-Hydro-Chemical MPM for Hydrate-Bearing Sediment
static Register<mpm::MPM, mpm::THCMPMExplicitHydrate<3>,
                const std::shared_ptr<mpm::IO>&>
    thc_mpm_explicit_hydrate_3d("THCMPMExplicitHydrate3D");

// 2D Explicit Thermo-Hydro-Mechanical-Chemical MPM for Hydrate-Bearing Sediment
static Register<mpm::MPM, mpm::THMCMPMExplicitHydrate<2>,
                const std::shared_ptr<mpm::IO>&>
    thmc_mpm_explicit_hydrate_2d("THMCMPMExplicitHydrate2D");

// 3D Explicit Thermo-Hydro-Mechanical-Chemical MPM for Hydrate-Bearing Sediment
static Register<mpm::MPM, mpm::THMCMPMExplicitHydrate<3>,
                const std::shared_ptr<mpm::IO>&>
    thmc_mpm_explicit_hydrate_3d("THMCMPMExplicitHydrate3D");