#include <iostream>
#include <string>
#include <Utils/StopWatch.hh>

#include <Algorithms/GradientDescent.hh>
#include <Algorithms/NewtonMethods.hh>  // Assuming NewtonMethod class exists
#include <Utils/OptimizationStatistic.hh>
#include <Utils/RandomNumberGenerator.hh>
#include <MassSpringSystemT.hh>
#include <Utils/DerivativeChecker.hh>

// Initialize different start points
std::vector<AOPT::GradientDescent::Vec> get_start_points(int n_grid_x, int n_grid_y) {
    std::vector<AOPT::GradientDescent::Vec> start_pts;

    const int n_vertices = (n_grid_x + 1) * (n_grid_y + 1);
    AOPT::RandomNumberGenerator rng2(-10., 10.);
    start_pts.push_back(rng2.get_random_nd_vector(2 * n_vertices));

    return start_pts;
}

int main(int _argc, const char* _argv[]) {
    if (_argc != 8) {
        std::cout << "Usage: input should be 'method(0: GradientDescent, 1: Newton), "
                  "function index(0: f without length, 1: f with length), "
                  "constrained spring scenario(1 or 2), "
                  "number of grid in x, number of grid in y, "
                  "max iteration, filename', e.g. "
                  "./MassSpringSolver 0 0 2 2 10000 /usr/spring" << std::endl;
        return -1;
    }

    // Read the input parameters
    int method, func_index, scenario, n_grid_x, n_grid_y, max_iter;
    method = atoi(_argv[1]);
    func_index = atoi(_argv[2]);
    scenario = atoi(_argv[3]);
    n_grid_x = atoi(_argv[4]);
    n_grid_y = atoi(_argv[5]);
    max_iter = atoi(_argv[6]);
    std::string filename(_argv[7]);

    // Construct mass spring system
    AOPT::MassSpringSystemT<AOPT::MassSpringProblem2DSparse> mss(n_grid_x, n_grid_y, func_index);
    mss.add_constrained_spring_elements(scenario);

    // Statistic instance
    auto opt_st = std::make_unique<AOPT::OptimizationStatistic>(mss.get_problem().get());

    // Initialize start points
    auto start_points = get_start_points(n_grid_x, n_grid_y);

    // Test on different start points
    for (auto i = 0u; i < start_points.size(); ++i) {
        // Set points
        mss.set_spring_graph_points(start_points[i]);

        // Initial energy
        auto energy = mss.initial_system_energy();
        std::cout << "\nInitial MassSpring system energy is " << energy << std::endl;

        // Save graph before optimization
        std::string fn = filename + std::string(std::to_string(i + 1));
        std::cout << "Saving initial spring graph to " << fn << "_*.csv" << std::endl;
        mss.save_spring_system(fn.c_str());

        // Start stopwatch
        Utils::StopWatch stopwatch;

        // Choose optimization method
        std::vector<double> x;
        if (method == 0) {  // Gradient Descent
            opt_st->start_recording();
            x = AOPT::GradientDescent::solve(opt_st.get(), start_points[i], 1e-4, max_iter);
            opt_st->print_statistics();
        } else if (method == 1) {  // Newton's Method
            opt_st->start_recording();
            x = AOPT::NewtonMethod::solve(opt_st.get(), start_points[i], 1e-4, max_iter);
            opt_st->print_statistics();
        }

        // Set the points after optimization
        mss.set_spring_graph_points(x);

        // Stop stopwatch and record time
        stopwatch.stop();
        std::cout << "Optimization completed in " << stopwatch.elapsed_seconds() << " seconds" << std::endl;

        // Save optimized graph
        fn += "_opt";
        std::cout << "Saving optimized spring graph to " << fn << "_*.csv" << std::endl;
        mss.save_spring_system(fn.c_str());
    }

    return 0;
}
