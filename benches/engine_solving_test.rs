use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, MatRef, Parallelism};
use faer_svd::{compute_svd, compute_svd_req, ComputeVectors, SvdParams};
use glam::{Quat, Vec3};
use nalgebra::{DVector, Dyn, Matrix, Matrix6xX, RowVector, Vector, Vector6, U1};
use ndarray::{Array1, Array2};
use nnls::nnls;

// add rotations later
const TARGETS: [EngineForce; 6] = [
    // forward
    EngineForce {
        force: Vec3::Z,
        torque: Vec3::ZERO,
    },
    // backward
    EngineForce {
        force: Vec3::NEG_Z,
        torque: Vec3::ZERO,
    },
    // right strafe
    EngineForce {
        force: Vec3::X,
        torque: Vec3::ZERO,
    },
    // left strafe
    EngineForce {
        force: Vec3::NEG_X,
        torque: Vec3::ZERO,
    },
    // upwards
    EngineForce {
        force: Vec3::Y,
        torque: Vec3::ZERO,
    },
    // downwards
    EngineForce {
        force: Vec3::NEG_Y,
        torque: Vec3::ZERO,
    },
];

#[derive(Debug, Clone, Copy)]
struct EngineForce {
    force: Vec3,
    torque: Vec3,
}

impl EngineForce {
    /// Relative to the center of mass
    fn at_point(force: Vec3, point: Vec3) -> Self {
        Self {
            force,
            torque: (point - Vec3::ZERO).cross(force),
        }
    }
}

fn generate_three_engines() -> Vec<(Vec3, Quat)> {
    Vec::from([
        (
            Vec3 {
                x: -8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 0.0,
                y: -8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
    ])
}

fn generate_four_engines() -> Vec<(Vec3, Quat)> {
    Vec::from([
        (
            Vec3 {
                x: -8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: -8.0,
                y: -8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 8.0,
                y: -8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
    ])
}

fn generate_five_engines() -> Vec<(Vec3, Quat)> {
    Vec::from([
        (
            Vec3 {
                x: -8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 8.0,
                y: 8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: -8.0,
                y: -8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 8.0,
                y: -8.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
        (
            Vec3 {
                x: 0.0,
                y: 0.0,
                z: -8.0,
            },
            Quat::IDENTITY,
        ),
    ])
}

fn generate_one_side(n: u8) -> Vec<(Vec3, Quat)> {
    assert!((3..=5).contains(&n));
    match n {
        3 => generate_three_engines(),
        4 => generate_four_engines(),
        5 => generate_five_engines(),
        _ => unreachable!(),
    }
}

/// Generates n engines on each side uniformly distributed across a side
fn make_input_data(n: u8) -> Vec<(Vec3, Quat)> {
    let mut result = Vec::new();
    let one_side = generate_one_side(n);
    // front
    result.extend(
        one_side
            .iter()
            .map(|(pos, _)| (pos, Quat::from_rotation_y(180.0_f32.to_radians())))
            .map(|(pos, rot)| (rot * *pos, rot))
            .collect::<Vec<_>>(),
    );
    // right
    result.extend(
        one_side
            .iter()
            .map(|(pos, _)| (pos, Quat::from_rotation_y(90.0_f32.to_radians())))
            .map(|(pos, rot)| (rot * *pos, rot))
            .collect::<Vec<_>>(),
    );
    // left
    result.extend(
        one_side
            .iter()
            .map(|(pos, _)| (pos, Quat::from_rotation_y(270.0_f32.to_radians())))
            .map(|(pos, rot)| (rot * *pos, rot))
            .collect::<Vec<_>>(),
    );
    // top
    result.extend(
        one_side
            .iter()
            .map(|(pos, _)| (pos, Quat::from_rotation_x(-90.0_f32.to_radians())))
            .map(|(pos, rot)| (rot * *pos, rot))
            .collect::<Vec<_>>(),
    );
    // bottom
    result.extend(
        one_side
            .iter()
            .map(|(pos, _)| (pos, Quat::from_rotation_x(90.0_f32.to_radians())))
            .map(|(pos, rot)| (rot * *pos, rot))
            .collect::<Vec<_>>(),
    );
    // back
    result.extend(one_side);
    result
}

fn generate_nalgebra_input(
    engines: Vec<(Vec3, Quat)>,
    target: EngineForce,
) -> (Matrix6xX<f32>, Vector6<f32>) {
    let ncols = engines.len();
    let engines_as_cols = engines.into_iter().flat_map(|(pos, rot)| {
        let engine_force = EngineForce::at_point(rot * Vec3::Z, pos);
        [
            engine_force.force.x,
            engine_force.force.y,
            engine_force.force.z,
            engine_force.torque.x,
            engine_force.torque.y,
            engine_force.torque.z,
        ]
    });
    let thrusts_matrix = Matrix6xX::from_iterator(ncols, engines_as_cols);
    let target_thrust = Vector6::from_iterator(
        [
            target.force.x,
            target.force.y,
            target.force.z,
            target.torque.x,
            target.torque.y,
            target.torque.z,
        ]
        .into_iter(),
    );
    (thrusts_matrix, target_thrust)
}

fn generate_faer_input(engines: Vec<(Vec3, Quat)>, target: EngineForce) -> (Mat<f32>, Mat<f32>) {
    let ncols = engines.len();
    let mut engines_as_cols = engines.into_iter().flat_map(|(pos, rot)| {
        let engine_force = EngineForce::at_point(rot * Vec3::Z, pos);
        [
            engine_force.force.x,
            engine_force.force.y,
            engine_force.force.z,
            engine_force.torque.x,
            engine_force.torque.y,
            engine_force.torque.z,
        ]
    });
    let thrusts_matrix = Mat::<f32>::with_dims(6, ncols, |_, _| engines_as_cols.next().unwrap());
    let target_thrust = Mat::<f32>::with_dims(6, 1, |row, _| match row {
        0 => target.force.x,
        1 => target.force.y,
        2 => target.force.z,
        3 => target.torque.x,
        4 => target.torque.y,
        5 => target.torque.z,
        _ => unreachable!(),
    });
    (thrusts_matrix, target_thrust)
}

fn solve_nnls(thrusts: &Matrix6xX<f32>, target: &Vector6<f32>) -> DVector<f32> {
    let result = nnls(
        Array2::from_shape_fn((6, thrusts.ncols()), |(i, j)| {
            *thrusts.get((i, j)).unwrap() as f64
        })
        .view(),
        Array1::from_shape_fn(6, |i| *target.get((i, 0)).unwrap() as f64).view(),
    )
    .0;
    DVector::from_iterator(thrusts.ncols(), result.into_iter().map(|n| n as f32))
}

fn solve_nalgebra_svd(thrusts: &Matrix6xX<f32>, target: &Vector6<f32>) -> DVector<f32> {
    let svd = thrusts.clone().svd(true, true);
    svd.solve(target, f32::EPSILON).unwrap()
}

fn solve_faer_svd(thrusts: MatRef<f32>, target: MatRef<f32>) -> Mat<f32> {
    assert_eq!(thrusts.nrows(), 6);
    assert_eq!(target.nrows(), 6);
    assert_eq!(target.ncols(), 1);

    let svd_req = compute_svd_req::<f32>(
        6,
        thrusts.ncols(),
        faer_svd::ComputeVectors::Full,
        ComputeVectors::Full,
        Parallelism::None,
        SvdParams::default(),
    )
    .unwrap();

    let mut mem = GlobalMemBuffer::new(svd_req);

    let mut stack = DynStack::new(&mut mem);

    let size = thrusts.nrows().min(thrusts.ncols());
    let mut s = Mat::zeros(size, 1);
    let mut u = Mat::zeros(thrusts.nrows(), size);
    let mut v = Mat::zeros(thrusts.ncols(), size);

    compute_svd(
        thrusts,
        s.as_mut().col(0),
        Some(u.as_mut()),
        Some(v.as_mut()),
        f32::EPSILON,
        f32::MIN_POSITIVE,
        Parallelism::None,
        stack.rb_mut(),
        SvdParams::default(),
    );

    let mut ut_b = u.adjoint() * target;

    for j in 0..ut_b.ncols() {
        let mut col = ut_b.as_mut().col(j);

        for i in 0..size {
            let val = s.read(i, 0);
            if val > f32::EPSILON {
                col.write(i, 0, col.read(i, 0) / val);
            } else {
                col.write(i, 0, 0.0);
            }
        }
    }

    v.transpose().adjoint() * ut_b
}

fn solve_aipia(thrusts: &Matrix6xX<f32>, target: &Vector6<f32>) -> DVector<f32> {
    let u_max = 1.0_f32;
    let mut b_clip = thrusts.clone();
    let mut u_full = DVector::<f32>::zeros(thrusts.ncols());

    let mut converged = false;

    let mut u_run = DVector::<f32>::zeros(thrusts.ncols());

    while !converged {
        let b_pseudo = b_clip.clone().pseudo_inverse(f32::EPSILON).unwrap();
        u_run = b_pseudo * (target - thrusts * &u_full);

        let mut fully_firing = 0;
        for i in 0..u_run.nrows() {
            if u_run[i] > u_max {
                fully_firing += 1;
                b_clip.fill_column(i, 0.0);
                u_full[i] = u_max;
            }
        }

        if fully_firing == 0 {
            let mut low_firing = 0;
            for i in 0..u_run.nrows() {
                let row_val = u_run[i];
                if row_val < -f32::EPSILON {
                    low_firing += 1;
                    b_clip.fill_column(i, 0.0);
                }
            }
            if low_firing == 0 {
                converged = true;
            }
        }

        if b_clip.iter().all(|e| e == &0.0) {
            converged = true
        }
    }

    u_full + u_run
}

fn bench_thrust_calculators(c: &mut Criterion) {
    let mut group = c.benchmark_group("Calculate coefficients (minimum L2 norm)");
    for n in 3..=5 {
        let thrusters = make_input_data(n);
        for target in TARGETS.iter() {
            let nalgebra_input = generate_nalgebra_input(thrusters.clone(), *target);
            group.bench_with_input(
                BenchmarkId::new(
                    "Adapted Iterative Pseudoinverse Algorithm",
                    format!(
                        "{} engines per side (force {}; torque {}) target force",
                        n, target.force, target.torque
                    ),
                ),
                &nalgebra_input,
                |b, i| b.iter(|| solve_aipia(&i.0, &i.1)),
            );
            /*
            group.bench_with_input(
                BenchmarkId::new(
                    "Nalgebra SVD",
                    format!(
                        "{} engines per side (force {}; torque {}) target force",
                        n, target.force, target.torque
                    ),
                ),
                &nalgebra_input,
                |b, i| b.iter(|| solve_nalgebra_svd(&i.0, &i.1)),
            );
            */
            /*
            let faer_input = generate_faer_input(thrusters.clone(), *target);
            group.bench_with_input(
                BenchmarkId::new(
                    "Faer SVD",
                    format!(
                        "{} engines per side (force {}; torque {}) target force",
                        n, target.force, target.torque
                    ),
                ),
                &faer_input,
                |b, i| b.iter(|| solve_faer_svd(i.0.as_ref(), i.1.as_ref())),
            );
            */
        }
    }
    group.finish();
}

criterion_group!(benches, bench_thrust_calculators);
criterion_main!(benches);
