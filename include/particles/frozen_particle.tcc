// Construct a two phase particle with id and coordinates
template <unsigned Tdim>
mpm::FrozenParticle<Tdim>::
     FrozenParticle(Index id, const VectorDim& coord)
    : mpm::Particle<Tdim>(id, coord) {
  this->initialise_liquid_phase();

  // Set material pointer to null
  liquid_material_ = nullptr;
  // Logger
  std::string logger =
      "FrozenParticle" + 
      std::to_string(Tdim) + "d::" + std::to_string(id);
  console_ = std::make_unique<spdlog::logger>(logger, mpm::stdout_sink);
}

//==============================================================================
// ASSIGN INITIAL CONDITIONS
//==============================================================================

// Initialise particle data from HDF5
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::initialise_particle(
    const HDF5Particle& particle) {
  // Derive from particle
  mpm::Particle<Tdim>::initialise_particle(particle);

  // Shared data
  this->liquid_material_id_ = particle.liquid_material_id;
  this->pore_pressure_ = particle.pore_pressure;
  // Liquid velocity
  Eigen::Vector3d liquid_velocity;
  liquid_velocity << particle.liquid_velocity_x, particle.liquid_velocity_y,
      particle.liquid_velocity_z;
  for (unsigned i = 0; i < Tdim; ++i)
    this->liquid_velocity_(i) = liquid_velocity(i);  

  // Liquid water data
  this->liquid_density_ = particle.liquid_density;
  this->liquid_mass_ = particle.liquid_mass;
  this->liquid_mass_density_ = particle.liquid_mass / particle.volume;
  this->pore_liquid_pressure_ = particle.pore_liquid_pressure;
  this->liquid_saturation_ = particle.liquid_saturation; 
  this->liquid_fraction_ = particle.liquid_fraction;
  this->viscosity_ = particle.viscosity; 
  this->permeability_ = particle.permeability;

  // Crystal ice data
  this->ice_density_ = particle.ice_density;
  this->ice_mass_ = particle.ice_mass;
  this->ice_mass_density_ = particle.ice_mass / particle.volume;
  this->pore_ice_pressure_ = particle.pore_ice_pressure;
  this->ice_fraction_ = particle.ice_fraction;  

  return true;
}

// Initialise particle HDF5 data and material
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::initialise_particle(
    const HDF5Particle& particle,
    const std::shared_ptr<mpm::Material<Tdim>>& material) {

  bool status = this->initialise_particle(particle);
  if (material != nullptr) {
    if (this->material_id_ == material->id() ||
        this->material_id_ == std::numeric_limits<unsigned>::max()) {
      material_ = material;
      // Reinitialize state variables
      auto mat_state_vars = material_->initialise_state_variables();
      if (mat_state_vars.size() == particle.nstate_vars) {
        unsigned i = 0;
        for (const auto& mat_state_var : mat_state_vars) {
          this->state_variables_[mat_state_var.first] = particle.svars[i];
          ++i;
        }
      }
    } else {
      status = false;
      throw std::runtime_error("Material is invalid to assign to particle!");
    }
  }
  return status;
}

// Initialise liquid phase particle properties
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::initialise_liquid_phase() {
  // Shared properties
  set_mixture_traction_ = false;
  set_contact_ = false;
  pore_pressure_ = 0.;
  PIC_pore_pressure_ = 0.;
  PIC_porosity_ = 0.;
  PIC_volume_ = 0.;
  PIC_volumetric_strain_ = 0.;
  PIC_deviatoric_strain_ = 0.;
  PIC_mean_stress_ = 0.;
  PIC_deviatoric_stress_ = 0.;

  // Liquid properties
  liquid_saturation_ = 1.;
  liquid_density_ = 0.;
  liquid_mass_ = 0.;
  liquid_mass_density_ = 0.;
  pore_liquid_pressure_ = 0.;
  liquid_fraction_ = 0.;
  liquid_velocity_.setZero();
  liquid_acceleration_.setZero();

  // Ice properties
  ice_density_ = 0.;
  ice_mass_ = 0.;
  ice_mass_density_ = 0.;
  pore_ice_pressure_ = 0.;
  pore_ice_pressure_increment_ = 0.;
  ice_fraction_ = 0;

  // Link data with NAME
  this->scalar_property_.insert({
    {"pore_pressures",         [&]() {return this->pore_pressure_;}},
    {"PIC_pore_pressures",     [&]() {return this->PIC_pore_pressure_;}},
    {"liquid_saturations",     [&]() {return this->liquid_saturation_;}},
    {"PIC_porosities",         [&]() {return this->PIC_porosity_;}},
    {"PIC_volumes",            [&]() {return this->PIC_volume_;}},
    {"liquid_fractions",       [&]() {return this->liquid_fraction_;}},
    {"liquid_densities",       [&]() {return this->liquid_density_;}},
    {"liquid_masses",          [&]() {return this->liquid_mass_;}},
    {"ice_fractions",          [&]() {return this->ice_fraction_;}},
    {"ice_saturations",        [&]() {return (1.0 - this->liquid_saturation_);}},
    {"ice_densities",          [&]() {return this->ice_density_;}},
    {"ice_masses",             [&]() {return this->ice_mass_;}},
    {"ice_mass_densities",     [&]() {return this->ice_mass_density_;}},
    {"permeabiities",          [&]() {return this->permeability_;}},
    {"PIC_volumetric_strains", [&]() {return this->PIC_volumetric_strain_;}},
    {"PIC_deviatoric_strains", [&]() {return this->PIC_deviatoric_strain_;}},
    {"mean_stresses",          [&]() {return this->PIC_mean_stress_;}},
    {"deviatoric_stresses",    [&]() {return this->PIC_deviatoric_stress_;}},
    {"deviatoric_strains",     [&]() {return this->deviatoric_strain_;}  }
  });

  this->vector_property_.insert({
    {"liquid_velocities",      [&]() {return this->liquid_velocity_;}},
    {"liquid_accelerations",   [&]() {return this->liquid_acceleration_;}}
  }); 
}

// Return particle data in HDF5 format
template <unsigned Tdim>
// cppcheck-suppress *
mpm::HDF5Particle mpm::FrozenParticle<Tdim>::hdf5() {
  // Derive from particle
  auto particle_data = mpm::Particle<Tdim>::hdf5();
  
  // Shared data
  // Particle liquid material id
  particle_data.liquid_material_id = this->liquid_material_id_; 
  // Particle liquid & ice velocity
  Eigen::Vector3d liquid_velocity;
  liquid_velocity.setZero();
  for (unsigned j = 0; j < Tdim; ++j)
    liquid_velocity[j] = this->liquid_velocity_[j];
  // Particle liquid velocity
  particle_data.liquid_velocity_x = liquid_velocity[0];
  particle_data.liquid_velocity_y = liquid_velocity[1];
  particle_data.liquid_velocity_z = liquid_velocity[2];
   // Particle pore pressure
  particle_data.pore_pressure = this->pore_pressure_; 

  // Liquid water data
  // Particle liquid density
  particle_data.liquid_density = this->liquid_density_;
  // Particle liquid mass
  particle_data.liquid_mass = this->liquid_mass_;
  // Particle pore water pressure
  particle_data.pore_liquid_pressure = this->pore_liquid_pressure_;  
  // Liquid water satauration
  particle_data.liquid_saturation = this->liquid_saturation_; 
  // Liquid fraction
  particle_data.liquid_fraction = this->liquid_fraction_;
  // Viscosity
  particle_data.viscosity = this->viscosity_; 
  // Permeability
  particle_data.permeability = this->permeability_;

  // Crystal ice data
  // Particle ice density
  particle_data.ice_density = this->ice_density_;
  // Particle ice mass
  particle_data.ice_mass = this->ice_mass_;
  // Particle pore ice pressure
  particle_data.pore_ice_pressure = this->pore_ice_pressure_;
  // Ice fraction
  particle_data.ice_fraction = this->ice_fraction_; 

  return particle_data;
}

// Assign a liquid material to particle
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_liquid_material(
    const std::shared_ptr<Material<Tdim>>& material) {
  bool status = false;
  try {
    // Check if material is valid and properties are set
    if (material != nullptr) {
      liquid_material_ = material;
      liquid_material_id_ = liquid_material_->id();
      status = true;
    } else {
      throw std::runtime_error("Material is undefined!");
    }

    // Assign permeability   
    this->assign_permeability();
    // Compute liquid/ice fraction

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return status;
}

// Assign particle permeability
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_permeability() {
  bool status = true;
  try {
    // Check if material ptr is valid
    if (material_ != nullptr) {
      // Porosity parameter
      permeability_ = material_->template property<double>("k_x");
    } else {
      throw std::runtime_error("Material is invalid");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assign degree of liquid water saturation (when considering phase change)
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_liquid_saturation_degree() {
  bool status = true;
  if (this->material_id_ != 999) {
  try {
    if (liquid_material_ != nullptr) {
      // Get freezing model
      int freezing_model = liquid_material_->template property<double>(std::string("freezing_model"));
      // Get critical temperature where melting happens
      double temperature_min = liquid_material_->template property<double>(std::string("temperature_min"));
      double temperature_max = liquid_material_->template property<double>(std::string("temperature_max"));
      this->porosity_ = material_->template property<double>(std::string("porosity"));      

      // Use temperature_, because it is smoother
      this->PIC_temperature_ = temperature_;

      double temperature = 0;
      if (PIC_temperature_ < temperature_min) temperature = temperature_min;
      else temperature = PIC_temperature_;
      // if (temperature_ < temperature_min) temperature = temperature_min;
      // else temperature = temperature_;

      // Compute initial liquid water saturation      
      // The function is effective within (temperature_min, temperature_max) 
      if (temperature < temperature_max) {
        switch (freezing_model) {
          // (Nishimura et al., 2009, Geotechnique)  
          case (1): {
            // Get lamda
            double lambda = liquid_material_->template property<double>(std::string("lambda"));
            // Get p0
            double p0 = liquid_material_->template property<double>(std::string("p0"));
            // get latent heat of fusion
            double latent_heat_of_fusion = liquid_material_->template property<double>(std::string("latent_heat_of_fusion"));  

            // Update liquid water saturation
            double A = -ice_density_ * latent_heat_of_fusion * log((temperature + 273.15) / 273.15);      
            double B = A / p0;
            double C =  std::pow(B, 1 / (1 - lambda));
            liquid_saturation_ =  std::pow((1 + C), -lambda);
            break;          
          }
          // (Huang et al., 2018, IJ-RM&MS)        
          case (2): {
            // Get lamda
            double lambda = liquid_material_->template property<double>(std::string("lambda"));
            liquid_saturation_ =  std::exp(lambda * temperature);
            break;           
          }
          // (McKenzie et al., 2007, WRR) 
          case (3): {
              // Get reference temperature 
            double temperature_ref = liquid_material_->template property<double>(std::string("temperature_ref"));
            double saturation_res = liquid_material_->template property<double>(std::string("saturation_res"));
            double freezing_point = liquid_material_->template property<double>(std::string("freezing_point"));
            liquid_saturation_ =  saturation_res + (1 - saturation_res) * 
                                        std::exp(-(std::pow((temperature - freezing_point) / temperature_ref, 2)));            
            break;          
          }
        } 
      } else liquid_saturation_ = 1;


      // Compute fraction of liquid water          
      this->liquid_fraction_ = liquid_saturation_ * porosity_;
      // COmpute fraction of ice
      this->ice_fraction_ = (1 - liquid_saturation_) * porosity_;  

      if (liquid_saturation_ < 0. || liquid_saturation_ > 1.)
        throw std::runtime_error(
            "Liquid water saturation degree is negative or larger than one");
    } else {
      throw std::runtime_error("Liquid material is invalid");
    }
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  }
  return status;
}

// Compute mass of particle (both solid and fluid)
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_mass() {
  mpm::Particle<Tdim>::compute_mass();
  this->compute_liquid_mass();
}

// Compute liquid mass of particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_liquid_mass() noexcept {
  // Check if particle volume is set and liquid material ptr is valid
  assert(volume_ != std::numeric_limits<double>::max() &&
         liquid_material_ != nullptr);

  this->liquid_density_ = 
    liquid_material_->template property<double>(std::string("density"));
  this->ice_density_ = 
    liquid_material_->template property<double>(std::string("ice_density"));

  // Liquid water mass
  this->liquid_mass_density_ = liquid_fraction_ * liquid_density_;
  this->liquid_mass_ = volume_ * liquid_mass_density_;
  
  // Crystal ice mass
  this->ice_mass_density_ = ice_fraction_ * ice_density_;
  this->ice_mass_ = volume_ * ice_mass_density_;
}

// Initial pore pressure
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::initialise_pore_pressure_watertable(
    const unsigned dir_v, const unsigned dir_h,
    std::map<double, double>& refernece_points) {
  bool status = true;
  try {
    // Initialise left boundary position (coordinate) and h0
    double left_boundary = std::numeric_limits<double>::lowest();
    double h0_left = 0.;
    // Initialise right boundary position (coordinate) and h0
    double right_boundary = std::numeric_limits<double>::max();
    double h0_right = 0.;
    // Position and h0 of particle (coordinate)
    const double position = this->coordinates_(dir_h);
    // Iterate over each refernece_points
    for (const auto& refernece_point : refernece_points) {
      // Find boundary
      if (refernece_point.first > left_boundary &&
          refernece_point.first <= position) {
        // Left boundary position and h0
        left_boundary = refernece_point.first;
        h0_left = refernece_point.second;
      } else if (refernece_point.first > position &&
                 refernece_point.first <= right_boundary) {
        // Right boundary position and h0
        right_boundary = refernece_point.first;
        h0_right = refernece_point.second;
      }
    }

    if (left_boundary != std::numeric_limits<double>::lowest()) {
      // Particle with left and right boundary
      if (right_boundary != std::numeric_limits<double>::max()) {
        this->pore_pressure_ =
            ((h0_right - h0_left) / (right_boundary - left_boundary) *
                 (position - left_boundary) +
             h0_left - this->coordinates_(dir_v)) *
            1000 * 9.81;
      } else
        // Particle with only left boundary
        this->pore_pressure_ =
            (h0_left - this->coordinates_(dir_v)) * 1000 * 9.81;
    }
    // Particle with only right boundary
    else if (right_boundary != std::numeric_limits<double>::max())
      this->pore_pressure_ =
          (h0_right - this->coordinates_(dir_v)) * 1000 * 9.81;

    else
      throw std::runtime_error(
          "Particle pore pressure can not be initialised by water table");
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

//==============================================================================
// APPLY BOUNDARY CONDITIONS
//==============================================================================

// Assign traction
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>:: assign_particle_traction(unsigned direction,
                                                          double traction) {
  bool status = true;
  this->assign_mixture_traction(direction, traction);
  return status;
}

// Assign traction to the mixture
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_mixture_traction(unsigned direction, 
                                                        double traction) {
  bool status = false;
  try {
    if (direction >= Tdim ||
        this->volume_ == std::numeric_limits<double>::max()) {
      throw std::runtime_error(
          "Particle mixture traction property: volume / direction is invalid");
    }
    // Assign mixture traction
    mixture_traction_(direction) =
        traction * this->volume_ / this->size_(direction);   
    status = true;
    this->set_mixture_traction_ = true;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Assign particle liquid phase velocity constraint
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_particle_liquid_velocity_constraint(
      unsigned dir, double velocity) {
  bool status = true;
  try {
    // Constrain directions can take values between 0 and Dim
    if (dir < Tdim)
      this->liquid_velocity_constraints_.insert(
          std::make_pair<unsigned, double>(static_cast<unsigned>(dir),
                                            static_cast<double>(velocity)));
    else
      throw std::runtime_error(
          "Particle liquid velocity constraint direction is out of bounds");

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Apply particle velocity constraints
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::apply_particle_liquid_velocity_constraints() {
  // Set particle velocity constraint
  for (const auto& constraint : this->liquid_velocity_constraints_) {
    // Direction value in the constraint (0, Dim)
    const unsigned dir = constraint.first;
    // Direction: dir % Tdim (modulus)
    const auto direction = static_cast<unsigned>(dir % Tdim);
    this->liquid_velocity_(direction) = constraint.second;
  }
}

// Assign particle pressure constraints
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_particle_pore_pressure_constraint(
      double pressure) {
  bool status = true;
  try {
    this->pore_pressure_constraint_ = pressure;
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Apply particle pore pressure constraints
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::apply_particle_pore_pressure_constraints(
      double pore_pressure) {
  // Set particle temperature constraint
  this->pore_pressure_ = pore_pressure;
}

//==============================================================================
// MAP PARTICLE INFORMATION TO NODES
//==============================================================================

// Map particle mass and momentum to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_mass_momentum_to_nodes() noexcept {
  if (this->material_id_ != 999) {                                       
    mpm::Particle<Tdim>::map_mass_momentum_to_nodes();
    this->map_ice_mass_momentum_to_nodes();
    this->map_liquid_mass_momentum_to_nodes();
  }
}

// Map ice mass and momentum to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_ice_mass_momentum_to_nodes() noexcept {
  // Check if ice mass is set and positive
  // assert(ice_mass_ != std::numeric_limits<double>::max());

  // Map ice mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Solid,
                  ice_mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Solid,
                  ice_mass_ * shapefn_[i] * velocity_);                     
  }
}

// Map liquid mass and momentum to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_liquid_mass_momentum_to_nodes() noexcept {
  // Check if liquid mass is set and positive
  assert(liquid_mass_ != std::numeric_limits<double>::max());

  // Map liquid mass and momentum to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_mass(true, mpm::ParticlePhase::Liquid,
                  liquid_mass_ * shapefn_[i]);
    nodes_[i]->update_momentum(true, mpm::ParticlePhase::Liquid,
                  liquid_mass_ * shapefn_[i] * liquid_velocity_); 
  }
}

// Map particle pore liquid pressure to nodes
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_pore_pressure_to_nodes(
    double current_time) noexcept {
  if (this->material_id_ != 999) {
    // Check if particle mass is set
    assert(liquid_mass_ != std::numeric_limits<double>::max());

    bool status = true;
    // Map particle liquid mass and pore pressure to nodes
    for (unsigned i = 0; i < nodes_.size(); ++i){
      nodes_[i]->update_mass_pressure(mpm::ParticlePhase::Liquid,
                                      shapefn_[i] * liquid_mass_ * pore_pressure_,
                                      current_time);    
    }
    return status;
  }
}

// Map body force for both mixture and liquid
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_external_force(
                                  const VectorDim& pgravity) {
  if (this->material_id_ != 999) {      
    this->map_mixture_body_force(mpm::ParticlePhase::Mixture, pgravity);
    this->map_liquid_body_force(pgravity);
  }
}

// Map liquid phase body force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_liquid_body_force(
                                const VectorDim& pgravity) noexcept {
  // Compute nodal liquid body forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, mpm::ParticlePhase::Liquid,
        (pgravity * liquid_mass_ * shapefn_(i)));
}

// Map mixture body force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_mixture_body_force(
                      unsigned mixture, const VectorDim& pgravity) noexcept {
  // Compute nodal mixture body forces
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodes_[i]->update_external_force(
        true, mixture, (pgravity * 
        (liquid_mass_ + ice_mass_ + this->mass_) * shapefn_(i)));        
}

// Map traction force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_traction_force() noexcept {
  if (this->material_id_ != 999)
    this->map_mixture_traction_force(mpm::ParticlePhase::Mixture);
}

// Map mixture traction force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_mixture_traction_force(
    unsigned mixture) noexcept {
  if (this->set_mixture_traction_) {
    // Map particle mixture traction forces to nodes
    for (unsigned i = 0; i < nodes_.size(); ++i)
      nodes_[i]->update_external_force(true, mixture,
                                       (shapefn_[i] * this->mixture_traction_));
  }
}

// Map both mixture and liquid internal force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_internal_force_semi (
                                                  double beta = 1) noexcept {
  if (this->material_id_ != 999) {
    mpm::FrozenParticle<Tdim>::map_mixture_internal_force(
        mpm::ParticlePhase::Mixture, beta);
    mpm::FrozenParticle<Tdim>::map_liquid_internal_force(beta);
  }
}

// Map liquid phase internal force
template <>
void mpm::FrozenParticle<2>::map_liquid_internal_force(double beta = 1) {

  unsigned ice_model = liquid_material_->template property<double>(std::string("ice_model"));                                    
  // initialise a vector of pore pressure
  Eigen::Matrix<double, 6, 1> pressure;
  pressure.setZero();
  pressure[0] = -this->pore_pressure_ * beta;
  pressure[1] = -this->pore_pressure_ * beta;

  if (is_axisymmetric_) pressure[2] = -this->pore_pressure_ * beta;

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;


    if (ice_model == 1 || ice_model == 3) {
      force[0] = dn_dx_(i, 0) * pressure[0] * liquid_fraction_;
      force[1] = dn_dx_(i, 1) * pressure[1] * liquid_fraction_;   
    }
    else if  (ice_model == 2) {
      force[0] = dn_dx_(i, 0) * pressure[0] * porosity_;
      force[1] = dn_dx_(i, 1) * pressure[1] * porosity_;   
    }

    if (is_axisymmetric_) 
      force[0] += shapefn_[i]/this->coordinates_(0) * pressure[2] * liquid_fraction_;
      force *= -1. * this->volume_;
      nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Liquid, force);
  }  

}

// Map liquid phase internal force
template <>
void mpm::FrozenParticle<3>::map_liquid_internal_force(double beta = 1) {
  // initialise a vector of pore pressure
  Eigen::Matrix<double, 6, 1> pressure;
  pressure.setZero();
  pressure[0] = -this->pore_pressure_ * beta;
  pressure[1] = -this->pore_pressure_ * beta;
  pressure[2] = -this->pore_pressure_ * beta;

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    force[0] = dn_dx_(i, 0) * pressure[0] * liquid_fraction_;
    force[1] = dn_dx_(i, 1) * pressure[1] * liquid_fraction_;
    force[2] = dn_dx_(i, 2) * pressure[2] * liquid_fraction_;

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mpm::ParticlePhase::Liquid, force);
  }
}

// Map mixture internal force
template <>
void mpm::FrozenParticle<2>::map_mixture_internal_force(unsigned mixture, 
                                                        double beta = 1) {
  // initialise a vector of pore pressure
  Eigen::Matrix<double, 6, 1> total_stress = this->stress_;

  unsigned ice_model = liquid_material_->template property<double>(std::string("ice_model"));
  if (ice_model == 1 || ice_model == 2) {
    total_stress[0] -= this->pore_pressure_ * beta * liquid_saturation_ +
                      this->pore_ice_pressure_ * (1.0 - liquid_saturation_);
    total_stress[1] -= this->pore_pressure_ * beta * liquid_saturation_ +
                      this->pore_ice_pressure_ * (1.0 - liquid_saturation_);
  } 
  else if (ice_model == 3) {
    total_stress[0] -= this->pore_pressure_ * beta;
    total_stress[1] -= this->pore_pressure_ * beta;
  }

  if (is_axisymmetric_) total_stress[2] -= 
                     this->pore_pressure_ * beta * liquid_saturation_ +
                     this->pore_ice_pressure_ * (1.0 - liquid_saturation_);

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 2, 1> force;
    force[0] = dn_dx_(i, 0) * total_stress[0] + dn_dx_(i, 1) * total_stress[3];
    force[1] = dn_dx_(i, 1) * total_stress[1] + dn_dx_(i, 0) * total_stress[3];
    
    if (is_axisymmetric_) 
      force[0] += shapefn_[i]/this->coordinates_(0) * total_stress[2];     

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mixture, force);

  }
}

// Map mixture internal force
template <>
void mpm::FrozenParticle<3>::map_mixture_internal_force(unsigned mixture, 
                                                        double beta = 1) {
  // initialise a vector of pore pressure
  Eigen::Matrix<double, 6, 1> total_stress = this->stress_;
  double cryo_suction = 0;
  cryo_suction = this->pore_ice_pressure_ - this->pore_pressure_ * beta;
  
  // if (ice_pressure) {
    double k_cryo = liquid_material_->template property<double>(std::string("k_cryo"));
  // } else {
  //   double k_cryo = 1 - liquid_saturation_;  
  // }


  total_stress[0] -= this->pore_pressure_ * beta * liquid_saturation_ +
                     this->pore_ice_pressure_ * (1.0 - liquid_saturation_) -
                     k_cryo * cryo_suction;
  total_stress[1] -= this->pore_pressure_ * beta * liquid_saturation_ +
                     this->pore_ice_pressure_ * (1.0 - liquid_saturation_) -
                     k_cryo * cryo_suction;
  total_stress[2] -= this->pore_pressure_ * beta * liquid_saturation_ +
                     this->pore_ice_pressure_ * (1.0 - liquid_saturation_) -
                     k_cryo * cryo_suction;                     

  // Compute nodal internal forces
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    // Compute force: -pstress * volume
    Eigen::Matrix<double, 3, 1> force;
    force[0] = dn_dx_(i, 0) * total_stress[0] + dn_dx_(i, 1) * total_stress[3] +
               dn_dx_(i, 2) * total_stress[5];

    force[1] = dn_dx_(i, 1) * total_stress[1] + dn_dx_(i, 0) * total_stress[3] +
               dn_dx_(i, 2) * total_stress[4];

    force[2] = dn_dx_(i, 2) * total_stress[2] + dn_dx_(i, 1) * total_stress[4] +
               dn_dx_(i, 0) * total_stress[5];

    force *= -1. * this->volume_;

    nodes_[i]->update_internal_force(true, mixture, force);
  }
}

// Map drag force
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_drag_force_coefficient() {
  if (this->material_id_ != 999) {
    try {
      // Update permeability
      this->update_permeability();
      // Initialise drag force coefficient
      double drag_force_coefficient;
      drag_force_coefficient = 0;

      // Check if permeability coefficient is valid
      if (permeability_ > 0.){
        // if viscosity / permeability     
        // drag_force_coefficient(i) = std::pow(liquid_fraction_, 2) * 
        //                             this->viscosity_ / permeability_(i);

        // if liquid density * gravity / hydraulic conductivity                                       
        drag_force_coefficient = std::pow(liquid_fraction_, 2) * 
                                    this->liquid_density_ * 9.81 / permeability_;                                            
      }else throw std::runtime_error("Permeability coefficient is invalid");

      // Map drag forces from particle to nodes
      for (unsigned j = 0; j < nodes_.size(); ++j)
        nodes_[j]->update_drag_force_coefficient(
            true, mpm::ParticlePhase::Liquid, 
            drag_force_coefficient * this->volume_ * shapefn_(j));

    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    }
  }
}

// Map particle heat capacity and heat to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_heat_to_nodes() {
  if (this->material_id_ != 999) {
    mpm::Particle<Tdim>::map_heat_to_nodes();
    this->map_liquid_heat_to_nodes();
  }
}

// Map liquid heat capacity and heat to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_liquid_heat_to_nodes() noexcept {
  // Check if liquid mass is set and positive
  assert(liquid_mass_ != std::numeric_limits<double>::max());
  // get the specific_heat of liquid       
  double liquid_specific_heat = liquid_material_->template
                      property<double>(std::string("specific_heat"));
  double ice_specific_heat = liquid_material_->template
                      property<double>(std::string("ice_specific_heat"));

  // compute total specific heat of liquid phase (liquid water + ice)
  double liquid_ice_heat_capacity = 
            liquid_mass_ * liquid_specific_heat + ice_mass_ * ice_specific_heat;

  // Map liquid heat capacity and heat to nodes
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    nodes_[i]->update_heat_capacity(true, mpm::ParticlePhase::Mixture,
                  liquid_ice_heat_capacity * shapefn_[i]);
    nodes_[i]->update_heat(true, mpm::ParticlePhase::Mixture,
                  liquid_ice_heat_capacity * shapefn_[i] * this->temperature_);      
  }
}

// Map particle heat conduction to node
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_heat_conduction() {
  if (this->material_id_ != 999) {                                              
    mpm::Particle<Tdim>::map_heat_conduction();
    mpm::FrozenParticle<Tdim>::map_liquid_heat_conduction();
  }    
}

// Map liquid phase heat conduction 
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_liquid_heat_conduction() noexcept {
  // Assign the liquid thermal conductivity
  double liquid_k_conductivity = liquid_material_->template
            property<double>(std::string("thermal_conductivity"));
  double ice_k_conductivity = liquid_material_->template
            property<double>(std::string("ice_thermal_conductivity"));

  // compute total thermal conductivity of liquid phase (liquid water + ice)
  double liquid_ice_k_conductivity =
                        liquid_fraction_ * liquid_k_conductivity +
                        ice_fraction_ * ice_k_conductivity;

  // Compute nodal heat conduction
  for (unsigned i = 0; i < nodes_.size(); ++i) {
    double heat_conduction = 0;
    for (unsigned j = 0; j < Tdim; ++j){
      heat_conduction += dn_dx_(i, j) * this->temperature_gradient_[j]; 
    }
    heat_conduction *= -1 * this->volume_ * liquid_ice_k_conductivity;
    nodes_[i]->update_heat_conduction(true, 
                            mpm::ParticlePhase::Mixture, heat_conduction);
  }
}

// Map heat convection of mixture
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_heat_convection() {
  if (this->material_id_ != 999) {
    // get the specific heat of liquid
    double liquid_specific_heat = 
          liquid_material_->template property<double>(std::string("specific_heat"));
    // Compute nodal heat convection
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      double heat_convection = 0;
      for (unsigned j = 0; j < Tdim; ++j){
        heat_convection += 
              shapefn_[i] * temperature_gradient_[j] * (liquid_velocity_[j] - velocity_[j]);
      }
      heat_convection *= -1 * liquid_mass_ * liquid_specific_heat;
      nodes_[i]->update_heat_convection(true, mpm::ParticlePhase::Mixture, heat_convection);
    }
  }
}

// Map latent heat of ice phase
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_latent_heat(double dt) noexcept {
  if (this->material_id_ != 999) {
    // get latent heat of fusion
    double latent_heat_of_fusion = liquid_material_->template
                        property<double>(std::string("latent_heat_of_fusion"));
    double latent_heat = dSl_dT_ * porosity_ * ice_density_ * latent_heat_of_fusion;          

    // Compute nodal latent heat
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      // latent_heat += shapefn_[i] * ice_fraction_derivative_; 
      nodes_[i]->update_latent_heat(true, mpm::ParticlePhase::Mixture, 
                                          volume_ * latent_heat * shapefn_[i]);
    }
  }
}

// Map convective heat flux at boundary
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_covective_heat_flux(
                                          double current_time) noexcept {
  if (this->material_id_ != 999) {
    bool covective_boundary = 
            material_->template property<bool>(std::string("covective_boundary"));

    if (covective_boundary) {
      if (id_ >= 1220 || id_ < 5) {
        // temperature_ = 0;
        double covective_coefficient = 
                material_->template property<double>(std::string("covective_coefficient"));
        // double outer_temperature = -1 * cos(2 * acos(-1) * current_time / 36.5);
        double outer_temperature = -1;
        double covective_heat = (outer_temperature - PIC_temperature_) * covective_coefficient;      
        // if (current_time > 2 ) outer_temperature = 1;     
        // Map particle heat source forces to nodes
        for (unsigned i = 0; i < nodes_.size(); ++i) {
          nodes_[i]->update_covective_heat_flux(true, mpm::ParticlePhase::Mixture,
                      (std::pow(this->volume_, 0.5) * shapefn_[i] * covective_heat)); 
        }                                
      }
    }
  }
}

//------------------------------------------------------------
// Semi-implict mpm
// Map local matrix of K_inter
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_K_inter_to_cell() {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {
      // Initialise multiplier
      VectorDim multiplier;
      multiplier.setZero();
      // Compute multiplier
      for (unsigned i = 0; i < Tdim; ++i){

        // // if viscosity / permeability
        // multiplier(i) = std::pow(liquid_fraction_, 2) * viscosity_ / permeability_(i);

        // if liquid density * gravity / hydraulic conductivity
        multiplier(i) = std::pow(liquid_fraction_, 2) * 
                                this->liquid_density_ * 9.81 / permeability_;
      }
      // Compute local matrix of K_inter
      cell_->compute_K_inter_element(shapefn_, volume_, multiplier);
    }
    catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Map laplacian matrix element
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_L_to_cell(
                                                    double dt, double alpha) {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {

        const double alpha_liquid = liquid_fraction_ * liquid_material_->template 
                            property<double>(std::string("compressibility")) +
                              ice_fraction_ * liquid_material_->template 
                            property<double>(std::string("ice_compressibility"));                    
        // calculate stablization parameter           
        const double youngs_modulus = material_->template 
                                  property<double>(std::string("youngs_modulus"));      
        const double poisson_ratio = material_->template 
                                  property<double>(std::string( "poisson_ratio"));
        const double stab_para = (1 + poisson_ratio) / youngs_modulus * 
            liquid_material_->template property<double>(std::string("stab_para"));
        // ice momentum model    
        const unsigned ice_model = liquid_material_->template property<double>(std::string("ice_model"));
        // Compute multiplier
        const double zeta = 1 - ice_density_ / liquid_density_;
        double multiplier_l = 0;
        double multiplier_s = 0;
      if (alpha) {
        const double drag = liquid_fraction_ * liquid_fraction_ * 9.81 * 
                      liquid_density_ / permeability_;                   
        const double ksi_l = (1 + dt * drag / 
                        liquid_fraction_ / (mass_density_ + ice_mass_density_)) / 
                        (1 + dt * drag * ((1 / liquid_mass_density_) + 
                        1 / (mass_density_ + ice_mass_density_))); 
        const double ksi_s = (1 + dt * drag / liquid_mass_density_ / 
                      (1 - liquid_fraction_)) / 
                      (1 + dt * drag * ((1 / liquid_mass_density_) + 
                        1 / (mass_density_ + ice_mass_density_)));

        multiplier_l = ksi_l * liquid_fraction_ / liquid_density_; 
        multiplier_s = ksi_s * 
                              (1 - liquid_fraction_ - zeta * ice_fraction_) *
                              (1 - liquid_fraction_) / 
                              (mass_density_ + ice_mass_density_);
  
      } else {
        multiplier_l = liquid_fraction_ / liquid_density_;
        if (ice_model == 1) {                            
          multiplier_s = (1 - liquid_fraction_ - zeta * ice_fraction_) *
                                (1 - porosity_) * liquid_saturation_ /
                                (mass_density_ + ice_mass_density_);
        }
        else if (ice_model == 2) {                            
          multiplier_s = (1 - liquid_fraction_ - zeta * ice_fraction_) *
                                (1 - porosity_) /
                                (mass_density_ + ice_mass_density_);
        }
        else if (ice_model == 3) {                            
          multiplier_s = (1 - liquid_fraction_ - zeta * ice_fraction_) *
                                (1 - liquid_fraction_) /
                                (mass_density_ + ice_mass_density_);
        }      
      } 
      // Compute local matrix of Laplacian
      cell_->compute_L_element(shapefn_, dn_dx_, volume_, multiplier_s, multiplier_l, alpha_liquid);
      cell_->compute_stab_element(shapefn_, shapefn_centroid_, stab_para);

    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Map element F_s_element, F_m_element
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_F_to_cell() {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {
      // multiplier_s * vs
      double zeta = 1 - ice_density_ / liquid_density_;
      double multiplier_s = 1 - zeta * ice_fraction_;
      // multiplier_l * （vl-vs）
      double multiplier_l = liquid_fraction_;
      // Compute local matrix of F
      cell_->compute_F_element(shapefn_, dn_dx_, volume_, multiplier_s, multiplier_l);
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Map element T_element
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_T_to_cell() {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {
      // get the thermal conductivity coefficient of solid, liquid, and ice 
      double beta_solid = material_->template
                        property<double>(std::string("thermal_expansivity"));  
      double beta_liquid = liquid_material_->template
                        property<double>(std::string("thermal_expansivity"));
      double beta_ice = liquid_material_->template
                        property<double>(std::string("ice_thermal_expansivity"));
    
      double beta = liquid_fraction_ * beta_liquid + ice_fraction_ * beta_ice + 
                    (1 - porosity_) * beta_solid;
      // double beta = 0;
      double zeta = 1 - this->ice_density_ / this->liquid_density_;
      // If considering the density difference of ice snd liquid
      beta -= zeta * dSl_dT_ * porosity_;                         

      // Compute local matrix of T
      cell_->compute_T_element(shapefn_, volume_, beta);

    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Map element T_element
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_P_to_cell(double beta) {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {
      double multiplier = 0;    
      if (!beta) {                    
        double alpha_liquid = liquid_fraction_ * liquid_material_->template 
                                  property<double>(std::string("compressibility"));
        multiplier = alpha_liquid * pore_pressure_; 
      }
      cell_->compute_P_element(shapefn_, volume_, multiplier);
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Map element K_cor_w_element_
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::map_K_cor_to_cell(
                                                  double dt, double alpha) {
  bool status = true;
  if (this->material_id_ != 999) {  
    try {
      // ice momentum model    
      const unsigned ice_model = liquid_material_->template property<double>(std::string("ice_model"));
      double multiplier_l = 0;
      double multiplier_s = 0;
      if (alpha) {
        double drag = liquid_fraction_ * liquid_fraction_ * 9.81 * 
                                              liquid_density_ / permeability_; 
        // bishop coefficient = ice_saturation
        double ksi_l = (1 + dt * drag / 
                        liquid_fraction_ / (mass_density_ + ice_mass_density_)) / 
                        (1 + dt * drag * ((1 / liquid_mass_density_) + 
                        1 / (mass_density_ + ice_mass_density_))); 
        double ksi_s = (1 + dt * drag / liquid_mass_density_ / (1 - liquid_fraction_)) / 
                        (1 + dt * drag * ((1 / liquid_mass_density_) + 
                        1 / (mass_density_ + ice_mass_density_)));

        multiplier_l = ksi_l * liquid_fraction_;
        multiplier_s = ksi_s * (1 - liquid_fraction_);                                       
      } else {
        if (ice_model == 1) {
          multiplier_l = liquid_fraction_;
          multiplier_s = (1 - porosity_) * liquid_saturation_;
        } 
        else if (ice_model == 2) {
          multiplier_l = porosity_;
          multiplier_s = 1 - porosity_;
        } 
        else if (ice_model == 3) {
          multiplier_l = liquid_fraction_;
          multiplier_s = 1 - liquid_fraction_;
        }      
      } 
      cell_->compute_K_cor_element(shapefn_, dn_dx_, volume_, multiplier_s, 
                          multiplier_l, this->coordinates_(0), is_axisymmetric_);    
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    }
  }
  return status;
}

//==============================================================================
// UPDATE PARTICLE INFORMATION
//==============================================================================

// Compute updated velocity and position of the particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_updated_velocity(
    double dt, double pic, double damping_factor) {
  mpm::Particle<Tdim>::compute_updated_velocity(dt, pic, damping_factor);
  this->compute_updated_liquid_velocity(dt, pic, damping_factor);
}

// Compute updated velocity of the liquid
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_updated_liquid_velocity(
    double dt, double pic, double damping_factor) {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  assert((-1.E-15) <= pic && pic <= (1 + 1.E-15));

  // Get PIC velocity
  Eigen::Matrix<double, Tdim, 1> pic_velocity =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < nodes_.size(); ++i)
    pic_velocity +=
        shapefn_[i] * nodes_[i]->velocity(mpm::ParticlePhase::Liquid);

  // Get interpolated nodal acceleration
  Eigen::Matrix<double, Tdim, 1> nodal_acceleration =
      Eigen::Matrix<double, Tdim, 1>::Zero();
  for (unsigned i = 0; i < nodes_.size(); ++i)
    nodal_acceleration +=
        shapefn_[i] * nodes_[i]->acceleration(mpm::ParticlePhase::Liquid);

  // Applying particle damping
  nodal_acceleration -= damping_factor * this->liquid_velocity_;
  this->liquid_acceleration_ = nodal_acceleration;

  // Get FLIP velocity
  Eigen::Matrix<double, Tdim, 1> flip_velocity =
      this->liquid_velocity_ + nodal_acceleration * dt;

  // Update particle velocity based on PIC value
  this->liquid_velocity_ = pic * pic_velocity + (1. - pic) * flip_velocity;  

  // Apply particle velocity constraints
  this->apply_particle_liquid_velocity_constraints();
}

// Map nodal pore pressure to particles
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::compute_updated_pore_pressure(double beta) {
  bool status = true;
  if (this->material_id_ != 999){  
    try {
      double pore_pressure_increment = 0.0;
      for (unsigned i = 0; i < nodes_.size(); ++i) {
        pore_pressure_increment +=
            shapefn_(i) * nodes_[i]->pore_pressure_increment();
      }
      // Get interpolated nodal pore pressure
      this->pore_pressure_ = pore_pressure_ * beta + pore_pressure_increment;

      // this->pore_pressure_ = 0.0;
      // for (unsigned i = 0; i < nodes_.size(); ++i) {
      //   this->pore_pressure_ +=
      //       shapefn_(i) * nodes_[i]->pressure(mpm::ParticlePhase::Liquid);
      // }

      // Apply free surface
      if (this->free_surface()) this->pore_pressure_ = 0.0;
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Compute pore liquid pressure smoothing based on nodal pressure
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::compute_pore_pressure_smoothing() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  bool status = true;
  if (this->material_id_ != 999){
    double pressure = 0;
    for (unsigned i = 0; i < nodes_.size(); ++i)
      pressure += shapefn_(i) * nodes_[i]->pressure(mpm::ParticlePhase::Liquid);

    // Update pore liquid pressure to interpolated nodal pressure
    this->pore_pressure_ = pressure;
    this->PIC_pore_pressure_ = pressure;
    // Apply free surface
    if (this->free_surface()) {
      this->pore_pressure_ = 0.0;
      this->PIC_pore_pressure_ = 0.0;
    }
  }
  return status;
}

// Compute pore liquid pressure and pore ice pressure at particles 
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::compute_liquid_ice_pore_pressure(double beta) {
  bool status = true;
  if (this->material_id_ != 999) {
    try {
      // get latent heat of fusion
      double latent_heat_of_fusion = liquid_material_->template
                          property<double>(std::string("latent_heat_of_fusion"));  

    if (PIC_temperature_ < 0) {
      pore_ice_pressure_ = ice_density_ / liquid_density_ * pore_pressure_ - 
                          ice_density_ * latent_heat_of_fusion * 
                          std::log((PIC_temperature_ + 273.15) / 273.15); 
    // if (pore_ice_pressure_ < 0) pore_ice_pressure_ = 0;
    } else pore_ice_pressure_ = 0;

    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
      status = false;
    }
  }
  return status;
}

// Update particle permeability
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_permeability() {
  if (this->material_id_ != 999) {  
    try {
      double porosity_0 = material_->template property<double>("porosity");
      double C = std::pow(this->porosity_, 3) / std::pow((1. - this->porosity_), 2) /
                 (std::pow(porosity_0, 3) / std::pow((1. - porosity_0), 2));
      // double C = 1; 
      permeability_ = C * material_->template property<double>("k_x");

      int permeability_model = liquid_material_->template property<double>(std::string("permeability_model"));
      double k_p = 1;
      if (permeability_model == 1) {
        // Get lambda
        double lambda = liquid_material_->template property<double>(std::string("lambda"));
        // Porosity parameter
        double A =  std::pow(liquid_saturation_, 0.5);
        double B =  std::pow(liquid_saturation_, 1 / lambda);
        double C =  std::pow(1 - B, lambda); 
        k_p =  A * std::pow((1 - C), 2);
        // if (k_p < 2E-7) k_p = 2E-7; 
        if (k_p < 1E-4) k_p = 1E-4;
      } else if (permeability_model == 2) {
        double omega = liquid_material_->template property<double>(std::string("omega"));
        k_p =  std::exp(- porosity_ * omega * (1 - liquid_saturation_) * log(10));
        if (k_p < 1E-6) k_p = 1E-6;
      }
      // Update permeability by KC equation
      permeability_ *= k_p; 
    
    } catch (std::exception& exception) {
      console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    }
  }
}

// Update liquid water saturation
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_liquid_saturation(double dt) {
  if (this->material_id_ != 999) {   
    // Get freezing model
    int freezing_model = liquid_material_->template property<double>(std::string("freezing_model"));  
    // Get temperature reference
    double temperature_min = liquid_material_->template property<double>(std::string("temperature_min"));
    double temperature_max = liquid_material_->template property<double>(std::string("temperature_max")); 

    double temperature = 0;
    if (PIC_temperature_ < temperature_min) temperature = temperature_min;
    else temperature = PIC_temperature_;

    // calculate liquid water saturation if temperature < temperature_max
    if (temperature < temperature_max) {
        // Compute initial liquid water saturation      
        switch (freezing_model) {
          // (Nishimura et al., 2009, Geotechnique)  
          case (1): {
            // Get lamda
            double lambda = liquid_material_->template property<double>(std::string("lambda"));
            // Get p0
            double p0 = liquid_material_->template property<double>(std::string("p0"));
            // get latent heat of fusion
            double latent_heat_of_fusion = liquid_material_->template property<double>(std::string("latent_heat_of_fusion")); 
            // liquid_saturation
            // double A = -(1 - ice_density_ / liquid_density_) * pore_pressure_ / p0
            double A = -ice_density_ * latent_heat_of_fusion * log((temperature + 273.15) / 273.15) / p0;    
            double B =  std::pow(A, 1 / (1 - lambda));
            liquid_saturation_ = std::pow((1 + B), -lambda);
            if (liquid_saturation_>=1) liquid_saturation_ =1;
            // dSl_dT
            double dSl_dB = -lambda * std::pow((1 + B), -lambda - 1);
            double dB_dA = 1 / (1 - lambda) * std::pow(A, 1 / (1 - lambda) - 1);
            double dA_dT = -ice_density_ * latent_heat_of_fusion / (temperature + 273.15) / p0;      
            dSl_dT_ = dSl_dB * dB_dA * dA_dT;
            if (PIC_temperature_ < temperature_min) dSl_dT_ = 0;
            // if (temperature_ < temperature_min) dSl_dT_ = 0;        
            break; 
          }        
          // (Huang et al., 2018, IJ-RM&MS)        
          case (2): {
            // Get lamda
            double lambda = liquid_material_->template property<double>(std::string("lambda"));
            // liquid_saturation
            liquid_saturation_ =  std::exp(lambda * temperature);
            // dSl_dT
            dSl_dT_ = lambda * std::exp(lambda * temperature);
            if (PIC_temperature_ < temperature_min) dSl_dT_ = 0;
            // if (temperature_ < temperature_min) dSl_dT_ = 0;
            break; 
          }
          // (McKenzie et al., 2007, WRR) 
          case (3): {
            // Get reference temperature 
            double temperature_ref = liquid_material_->template property<double>(std::string("temperature_ref"));
            double saturation_res = liquid_material_->template property<double>(std::string("saturation_res"));
            double freezing_point = liquid_material_->template property<double>(std::string("freezing_point"));          
            // liquid_saturation          
            liquid_saturation_ =  saturation_res + (1 - saturation_res) * 
                                        std::exp(-(std::pow((temperature - freezing_point) / temperature_ref, 2)));
            // dSl_dT
            dSl_dT_ = (1 - saturation_res) * std::exp(-(std::pow((temperature - freezing_point) / temperature_ref, 2))) *
                      (-2 * (temperature - freezing_point) / temperature_ref / temperature_ref);
            if (PIC_temperature_ < temperature_min) dSl_dT_ = 0;
            // if (temperature_ < temperature_min) dSl_dT_ = 0;                             
            break;        
          }
        }
      } else {
        liquid_saturation_ = 1.0;
        dSl_dT_ = 0;
      }

    double ice_fraction_k = this->ice_fraction_;
    // Ice fraction at time k+1  
    this->ice_fraction_ = (1 - liquid_saturation_) * porosity_;
    // Liquid fraction at time k+1  
    this->liquid_fraction_ = liquid_saturation_ * porosity_;
    ice_fraction_increment_ = this->ice_fraction_ - ice_fraction_k;
  }      
}

// update density of the particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_particle_density(double dt) {
  if (this->material_id_ != 999) { 
    mpm::Particle<Tdim>::update_particle_density(dt);
    this->update_liquid_density();
  }
}

// update liquid density of the particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_liquid_density() noexcept {
  // Check if liquid material ptr is valid
  assert(liquid_material_ != nullptr);
  // get initial density of liquid and ice
  double liquid_density_0 = liquid_material_->template
                    property<double>(std::string("density"));
  double ice_density_0 = liquid_material_->template
                    property<double>(std::string("ice_density"));

  // get the compressibility of liquid and ice
  double alpha_liquid = liquid_material_->template
                    property<double>(std::string("compressibility"));
  double alpha_ice = liquid_material_->template
                    property<double>(std::string("ice_compressibility"));

  // get the thermal conductivity coefficient of liquid and ice
  double beta_liquid = liquid_material_->template
                    property<double>(std::string("thermal_expansivity"));
  double beta_ice = liquid_material_->template
                    property<double>(std::string("ice_thermal_expansivity"));


  // Intrinsic density is a function of temperature and pressure
  this->liquid_density_ = liquid_density_0 *
                    std::exp(pore_pressure_ * alpha_liquid - 
                    3 * beta_liquid * PIC_temperature_);   
  this->ice_density_ = ice_density_0 *
                    std::exp(-3 * beta_ice * PIC_temperature_);                                  
}

// update mass of the particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_particle_volume()  {   
  if (this->material_id_ != 999) {
    mpm::Particle<Tdim>::update_particle_volume();  
    this->update_liquid_mass();
  }  
}

// update liquid mass of particle
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_liquid_mass() noexcept {
  // Check if particle volume is set and liquid material ptr is valid
  assert(volume_ != std::numeric_limits<double>::max() &&
         liquid_material_ != nullptr);

  // Liquid Mass = volume of particle * liquid_mass_density
  this->liquid_mass_density_ = liquid_fraction_ * liquid_density_;
  this->liquid_mass_ = volume_ * liquid_mass_density_;

  // Ice Mass = volume of particle * ice_mass_density
  this->ice_mass_density_ = ice_fraction_ * ice_density_;
  this->ice_mass_ = volume_ * ice_mass_density_;
}

// Update viscosity
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::update_viscosity() noexcept {
  // this->viscosity_ = 2.1 * 1E-6 * exp(1808.5 / (273.15 + PIC_temperature_)); 
  this->viscosity_ = 1.793E-3;
}

// Compute thermal strain of the ice
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_ice_thermal_strain() noexcept {
  if (this->material_id_ != 999) {
    // get the thermal conductivity coefficient
    double beta_solid =
      material_->template property<double>(std::string("thermal_expansivity"));
    double beta_ice =
      liquid_material_->template property<double>(std::string("ice_thermal_expansivity"));

    double beta_ice_solid = (ice_fraction_ * beta_ice + (1 - porosity_) * beta_solid)/
                              (1 - liquid_fraction_); 
                              
    // compute thermal strain increment
    for (unsigned i = 0; i < 3; i++) {
      dthermal_strain_[i] = -beta_ice_solid * this->temperature_increment_;
    }

    // Compute volumetric thermal strain
    dthermal_volumetric_strain_ = dthermal_strain_.head(Tdim).sum();
    if (is_axisymmetric_) {
      dthermal_volumetric_strain_ = dthermal_strain_.head(3).sum();    
    }
    
    // update thermal strain 
    thermal_strain_ += dthermal_strain_;
    thermal_volumetric_strain_ += dthermal_volumetric_strain_;
  }
}

// Compute frost heave 
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_frost_heave_strain() noexcept {
  if (this->material_id_ != 999) {
    // compute strain increment caused by frost heave
    double heave_coeff = (liquid_density_ - ice_density_) / liquid_density_ / 3;
    for (unsigned i = 0; i < 3; i++) {
      //dheave_strain_[i] =  -heave_coeff * ice_fraction_increment_ *
                          // ice_fraction_ / (1 - liquid_fraction_)  ;;
      dheave_strain_[i] = heave_coeff * dSl_dT_ * porosity_ * temperature_increment_ *
                          ice_fraction_ / (1 - liquid_fraction_)  ;
    }

    // Compute volumetric thermal strain
    dheave_volumetric_strain_ = dheave_strain_.head(Tdim).sum();
    if (is_axisymmetric_) {
      dheave_volumetric_strain_ = dheave_strain_.head(3).sum();
    }
    dthermal_strain_ += dheave_strain_;
    dthermal_volumetric_strain_ += dheave_volumetric_strain_;
    // update thermal strain 
    thermal_strain_ += dheave_strain_;
    thermal_volumetric_strain_ += dheave_volumetric_strain_; 
  }
}

// Compute pore pressure
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::compute_pore_pressure(double dt) {
  // Check if liquid material and cell pointer are set and positive
  assert(liquid_material_ != nullptr && cell_ != nullptr);
  if (this->material_id_ != 999){
    // get the bulk modulus of liquid
    const double K =
      liquid_material_->template property<double>(std::string("bulk_modulus"));
    // get the thermal conductivity coefficient of solid, liquid, and ice 
    const double beta_solid = material_->template
                      property<double>(std::string("thermal_expansivity"));
    const double beta_liquid = liquid_material_->template
                      property<double>(std::string("thermal_expansivity"));
    const double beta_ice = liquid_material_->template
                      property<double>(std::string("ice_thermal_expansivity"));
    const double beta_m = liquid_fraction_ * beta_liquid + ice_fraction_ * beta_ice + 
                 (1 - porosity_) * beta_solid;

    const double zeta = 1 - ice_density_ / liquid_density_;
    const double beta = zeta * porosity_ * dSl_dT_ - beta_m;

    // Compute at centroid
    // get liquid phase strain rate at cell centre
    auto liquid_strain_rate =
        this->compute_strain_rate(dn_dx_centroid_, shapefn_centroid_, mpm::ParticlePhase::Liquid);

    // update pressure
    this->pore_pressure_ += -dt * (K / liquid_fraction_) *
                  ((1 - liquid_fraction_ - zeta * ice_fraction_) * strain_rate_.head(Tdim).sum() +
                  liquid_fraction_ * liquid_strain_rate.head(Tdim).sum()) - 
                  beta * temperature_increment_;
  }
}

// Map particle pore liquid pressure to nodes
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_scalers_to_nodes() {
  if (this->material_id_ != 999) {  
    // Check if particle mass is set
    assert(liquid_mass_ != std::numeric_limits<double>::max());
    if (this->material_id_ != 999){
      if (Tdim == 2) {
        // Compute deviatoric strain
        const double volumetric_strain = this->strain_.head(2).sum();
        Eigen::Matrix<double, 6, 1> strain_dev = this->strain_;
        strain_dev.head(2).noalias() += 
                          -Eigen::Vector2d::Constant(volumetric_strain / 2);
        const double deviatoric_strain = std::pow(2. * 
                            (strain_dev.dot(strain_dev) + 
                            strain_dev.tail(3).dot(strain_dev.tail(3))), 0.5);
        // Compute deviatoric stress
        const double mean_stress = this->stress_.head(2).sum() / 2.0;
        Eigen::Matrix<double, 6, 1> stress_dev = this->stress_;
        stress_dev.head(2).noalias() += 
                              -Eigen::Vector2d::Constant(mean_stress);
        const double deviatoric_stress = std::pow(0.5 * 
                            (stress_dev.dot(stress_dev) + 
                            stress_dev.tail(3).dot(stress_dev.tail(3))), 0.5);
        this->deviatoric_strain_ = deviatoric_strain;
        // Map particle liquid mass and pore pressure to nodes
        for (unsigned i = 0; i < nodes_.size(); ++i) {
          nodes_[i]->update_scalers(true, 0, shapefn_[i] * 1.0);
          nodes_[i]->update_scalers(true, 1, shapefn_[i] * 1.0 * porosity_);
          nodes_[i]->update_scalers(true, 2, shapefn_[i] * 1.0 * volume_);
          nodes_[i]->update_scalers(true, 3, shapefn_[i] * 1.0 * liquid_saturation_);
          nodes_[i]->update_scalers(true, 4, shapefn_[i] * 1.0 * volumetric_strain);
          nodes_[i]->update_scalers(true, 5, shapefn_[i] * 1.0 * deviatoric_strain);
          nodes_[i]->update_scalers(true, 6, shapefn_[i] * 1.0 * mean_stress);
          nodes_[i]->update_scalers(true, 7, shapefn_[i] * 1.0 * deviatoric_stress);
        }
      }
    }
  }
}

// Compute pore liquid pressure smoothing based on nodal pressure
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::compute_scalers_smoothing() noexcept {
  // Check if particle has a valid cell ptr
  assert(cell_ != nullptr);
  bool status = true;  
  if (this->material_id_ != 999) {
    double porosity = 0;
    double volume = 0;
    double liquid_saturation = 0;
    double volumetric_strain = 0;
    double deviatoric_strain = 0;
    double mean_stress = 0;
    double deviatoric_stress = 0;
    for (unsigned i = 0; i < nodes_.size(); ++i) {
      porosity += shapefn_(i) * nodes_[i]->smoothed_scalers(1);
      volume += shapefn_(i) * nodes_[i]->smoothed_scalers(2);
      liquid_saturation += shapefn_(i) * nodes_[i]->smoothed_scalers(3);
      volumetric_strain += shapefn_(i) * nodes_[i]->smoothed_scalers(4);
      deviatoric_strain += shapefn_(i) * nodes_[i]->smoothed_scalers(5);
      mean_stress += shapefn_(i) * nodes_[i]->smoothed_scalers(6);
      deviatoric_stress += shapefn_(i) * nodes_[i]->smoothed_scalers(7);
    }

    // Update pore liquid pressure to interpolated nodal pressure
    this->PIC_porosity_ = porosity; 
    this->PIC_volume_ = volume; 
    this->PIC_liquid_saturation_ = liquid_saturation;
    this->PIC_volumetric_strain_ = volumetric_strain; 
    this->PIC_deviatoric_strain_ = deviatoric_strain; 
    this->PIC_mean_stress_ = mean_stress;
    this->PIC_deviatoric_stress_ = deviatoric_stress;
  }
  return status;
}

//==============================================================================
// CONTACT
//==============================================================================

// Assign contact to the particle
template <unsigned Tdim>
bool mpm::FrozenParticle<Tdim>::assign_particle_contact(unsigned dir, 
                                                        double normal) {
  bool status = false;
  try {
    // Assign contact
    contact_normal_(dir) = normal;
    status = true;
    this->set_contact_ = true;
    for (unsigned i = 0; i < 4; ++i) {       
      nodes_[i]->update_contact_normal(shapefn_[i] * this->mass_,
                             shapefn_[i] * this->mass_ * contact_normal_);
    }   
  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
    status = false;
  }
  return status;
}

// Overwrite node velocity to get strain correct
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_moving_rigid_velocity_to_nodes(
    unsigned dir, double velocity, double dt) noexcept {   
  if ((this->material_id_ == 999) & (id_ > 240)){  
    for (unsigned i = 0; i < 4; ++i) {
      nodes_[i]->assign_velocity_from_rigid(dir, velocity, dt);
    }       
  }
}

// Overwrite node velocity to get strain correct
template <unsigned Tdim>
void mpm::FrozenParticle<Tdim>::map_rigid_mass_momentum_to_nodes() noexcept {   
  if (this->material_id_ == 999 && this->set_contact_){  
    for (unsigned i = 0; i < 4; ++i) {       
      nodes_[i]->update_rigid_mass_momentum(shapefn_[i] * this->mass_, 
                                  shapefn_[i] * this->mass_ * velocity_);     
    }       
  }
}
