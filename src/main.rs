use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use dyn_stack::{DynStack, GlobalMemBuffer, ReborrowMut};
use faer_core::{Mat, MatRef, Parallelism};
use faer_svd::{compute_svd, compute_svd_req, ComputeVectors, SvdParams};
use glam::{Quat, Vec3};
use nalgebra::{DVector, Matrix6xX, Vector6};
use ndarray::{Array1, Array2};
use nnls::nnls;

const TARGET: EngineForce = EngineForce {
    force: Vec3::NEG_Y,
    torque: Vec3::ZERO,
};

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
            .map(|(pos, _)| (pos, Quat::from_rotation_x(270.0_f32.to_radians())))
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

fn solve_nalgebra_svd(thrusts: &Matrix6xX<f32>, target: &Vector6<f32>) -> DVector<f32> {
    let svd = thrusts.clone().svd(true, true);
    svd.solve(target, f32::EPSILON).unwrap()
}

const ENGINES: u8 = 4;

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

fn main() {
    let mut target = TARGET;
    target.force *= 4.0;
    let engines = make_input_data(ENGINES);
    let input = generate_nalgebra_input(engines, target);
    let result = solve_nnls(&input.0, &input.1);
    println!("{}", result);
}
