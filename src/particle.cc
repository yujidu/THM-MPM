#include "particle.h"
#include "factory.h"
#include "particle_base.h"
#include "twophase_particle.h"
#include "threephase_particle.h"
#include "frozen_particle.h"
#include "frozen_particle_unsaturated.h"
#include "hydrate_particle.h"

// Single phase particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::Particle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2d("P2D");

// Single phase particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::Particle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3d("P3D");

// Two phase particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::TwoPhaseParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2d2phase("P2D2PHASE");

// Two phase particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::TwoPhaseParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3d2phase("P3D2PHASE");

// Three phase particle2D (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::ThreePhaseParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2d3phase("P2D3PHASE");

// Three phase particle3D (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::ThreePhaseParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3d3phase("P3D3PHASE");

// Two phase particle2D with phase change (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::FrozenParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dsatfrozen("P2DFROZENSAT");

// Two phase particle3D with phase change (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::FrozenParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dsatfrozen("P3DFROZENSAT");

// Three phase particle2D with phase change (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::UnsatFrozenParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dunsatfrozen("P2DFROZENUNSAT");

// Three phase particle3D with phase change (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::UnsatFrozenParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dunsatfrozen("P3DFROZENUNSAT");

// Two phase particle2D with phase change (2 Dim)
static Register<mpm::ParticleBase<2>, mpm::HydrateParticle<2>, mpm::Index,
                const Eigen::Matrix<double, 2, 1>&>
    particle2dhydrate("P2DHYDRATE");

// Two phase particle3D with phase change (3 Dim)
static Register<mpm::ParticleBase<3>, mpm::HydrateParticle<3>, mpm::Index,
                const Eigen::Matrix<double, 3, 1>&>
    particle3dhydrate("P3DHYDRATE");