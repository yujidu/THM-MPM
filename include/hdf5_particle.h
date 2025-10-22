#ifndef MPM_HDF5_H_
#define MPM_HDF5_H_

// HDF5
#include "hdf5.h"
#include "hdf5_hl.h"

#include "data_types.h"

namespace mpm {
// Define a struct of particle
typedef struct HDF5Particle {
  // Index
  mpm::Index id;
  // Index 
  mpm::Index cell_id;
  // Material id
  unsigned material_id;
  // Liquid material id
  unsigned liquid_material_id;
  // Current time
  double current_time;
  // Mass
  double mass;
  // Liquid mass
  double liquid_mass;
  // Ice mass
  double ice_mass; 
  // Hydrate mass
  double hydrate_mass;
  // Gas mass
  double gas_mass;   
  // Volume
  double volume;
  // Density
  double density;
  // Liquid density
  double liquid_density;  
  // Ice density
  double ice_density; 
  // Hydrate density
  double hydrate_density;  
  // Gas density
  double gas_density;   
  // Porosity
  double porosity;
  // Liquid water saturation
  double liquid_saturation;
  // Ice saturation
  double ice_saturation;  
  // Hydrate saturation
  double hydrate_saturation;
  // Gas saturation
  double gas_saturation;  
  // Liquid fraction
  double liquid_fraction;
  // Ice fraction
  double ice_fraction;
  // Hydrate fraction
  double hydrate_fraction;
  // Gas fraction
  double gas_fraction;
  // viscosity
  double viscosity;
  // Permeability
  double permeability;
  // Permeability
  double liquid_permeability;
   // Permeability
  double gas_permeability;     
  // Pressure
  double pressure;
  // Pore pressure
  double pore_pressure;
  // Pore pressure
  double pore_liquid_pressure;
  // Pore pressure
  double pore_ice_pressure;
  // Pore pressure
  double liquid_pressure;
  // Pore pressure
  double gas_pressure;
  // Temperature
  double temperature;
  // Temperature
  double PIC_temperature;  
  // Coordinates
  double coord_x, coord_y, coord_z;
  // Displacement
  double displacement_x, displacement_y, displacement_z;
  // Natural particle size
  double nsize_x, nsize_y, nsize_z;
  // Velocity
  double velocity_x, velocity_y, velocity_z;
  // Liquid velocity
  double liquid_velocity_x, liquid_velocity_y, liquid_velocity_z;
  // Gas velocity
  double gas_velocity_x, gas_velocity_y, gas_velocity_z;  
  // Stresses
  double stress_xx, stress_yy, stress_zz;
  double tau_xy, tau_yz, tau_xz;
  // Strains
  double strain_xx, strain_yy, strain_zz;
  double gamma_xy, gamma_yz, gamma_xz;
  // Thermal strains
  double thermal_strain_xx, thermal_strain_yy, thermal_strain_zz;
  double thermal_gamma_xy, thermal_gamma_yz, thermal_gamma_xz;
  // Liquid strain
  double liquid_strain_xx, liquid_strain_yy, liquid_strain_zz;
  double liquid_gamma_xy, liquid_gamma_yz, liquid_gamma_xz;
  // Volumetric strain centroid
  double epsilon_v;
  // Thermal volumetric strain centroid
  double thermal_epsilon_v;
  // Deformation gradient
  double fxx,fxy,fxz,fyx,fyy,fyz,fzx,fzy,fzz;
  // Fabric
  double fabric;
  // Rotation
  double rotation_xy, rotation_yz, rotation_xz;
  // State variables (init to zero)
  double svars[20] = {0};
  // Status
  bool status;
  // Number of state variables
  unsigned nstate_vars;  
} HDF5Particle;

namespace hdf5::particle {
const hsize_t NFIELDS = 102;

const size_t dst_size = sizeof(HDF5Particle);

// Destination offset
extern const size_t dst_offset[NFIELDS];

// Destination size
extern const size_t dst_sizes[NFIELDS];

// Define particle field information
extern const char* field_names[NFIELDS];

// Initialise field types
extern const hid_t field_type[NFIELDS];

}  // namespace hdf5::particle

}  // namespace mpm

#endif  // MPM_HDF5_H_
