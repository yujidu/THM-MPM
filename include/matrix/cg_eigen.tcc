//! Conjugate Gradient Solver
template <unsigned Tdim>
Eigen::VectorXd mpm::CGEigen<Tdim>::solve(const Eigen::SparseMatrix<double>& A,
                                          const Eigen::VectorXd& b,
                                          std::string solver_type,
                                          int num_threads) {
  Eigen::VectorXd x;
  Eigen::initParallel();
  omp_set_num_threads(num_threads);
  Eigen::setNbThreads(num_threads); 

  try {         
    if (solver_type == "cg") {
      
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;

      solver.setMaxIterations(max_iter_);
      solver.setTolerance(tolerance_);
      solver.compute(A);

      x = solver.solve(b);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Fail to solve Poisson equation\n");
      }

    } else if (solver_type == "lscg") {

      // Another option is LDLT, but not accurate as lscg
      // Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
      Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;

      solver.setMaxIterations(max_iter_);
      solver.setTolerance(tolerance_);
      solver.compute(A);

      x = solver.solve(b);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Fail to solve intermediate velocity\n");
      }
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return x;
}

//! Conjugate Gradient Solver
template <unsigned Tdim>
Eigen::VectorXd mpm::CGEigen<Tdim>::solve(
    const Eigen::SparseMatrix<double>& A, const Eigen::VectorXd& b,
    std::string solver_type, const Eigen::VectorXd& initial_guess) {
  Eigen::VectorXd x;
  try {

    if (solver_type == "cg") {
      Eigen::ConjugateGradient<Eigen::SparseMatrix<double>> solver;

      solver.setMaxIterations(max_iter_);
      solver.setTolerance(tolerance_);
      solver.compute(A);

      x = solver.solveWithGuess(b, initial_guess);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Fail to solve Poisson equation\n");
      }

    } else if (solver_type == "lscg") {

      Eigen::LeastSquaresConjugateGradient<Eigen::SparseMatrix<double>> solver;

      solver.setMaxIterations(max_iter_);
      solver.setTolerance(tolerance_);
      solver.compute(A);
      x = solver.solveWithGuess(b, initial_guess);

      if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Fail to solve intermediate acceleration\n");
      }
    }

  } catch (std::exception& exception) {
    console_->error("{} #{}: {}\n", __FILE__, __LINE__, exception.what());
  }
  return x;
}

/*
//! Precondition Jacobian
template <unsigned Tdim>
typename mpm::CGEigen<Tdim>::Eigen::VectorXd
    mpm::CGEigen<Tdim>::precondition_jacobian() {
  const size_t n = vec_b_->size();
  Eigen::VectorXd vm(n);

  vm.setZero();
  for (auto& mat_a : *mat_a_) {
    for (size_t i = 0; i < n; ++i) {
      vm[i] += mat_a.coeff(i, i);
    }
  }
  for (size_t i = 0; i < n; ++i) {
    vm[i] = 1 / vm[i];
    // When beta is zero, the stiffness matrix will have zero value elements
    if (!std::isfinite(vm[i])) vm[i] = 1.0;
  }
  vm = vm.array() * vrestraints_.array();
  return vm;
}

//! Cholesky solver
template <unsigned Tdim>
bool mpm::CGEigen<Tdim>::cholesky() {
  SparseMatrix stiff;
  for (auto& mat_a : *mat_a_) stiff += mat_a;

  Eigen::SimplicialCholesky<SparseMatrix> cholesky(stiff);
  *vec_x_ = cholesky.solve(*vec_b_);
  return (cholesky.info() == Eigen::Success);
}
*/
