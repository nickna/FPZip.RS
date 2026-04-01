#[allow(dead_code)]
mod test_helpers;

use test_helpers::{generate_double_field, generate_float_field};

#[test]
fn lcg_first_iteration_produces_expected_seed() {
    let expected = (1103515245u32.wrapping_mul(1).wrapping_add(12345)) & 0x7FFFFFFF;
    assert_eq!(expected, 1103527590);
}

#[test]
fn float_field_deterministic() {
    let field1 = generate_float_field(10, 10, 10, 0.0, 1);
    let field2 = generate_float_field(10, 10, 10, 0.0, 1);
    assert_eq!(field1, field2);
}

#[test]
fn double_field_deterministic() {
    let field1 = generate_double_field(10, 10, 10, 0.0, 1);
    let field2 = generate_double_field(10, 10, 10, 0.0, 1);
    assert_eq!(field1, field2);
}

#[test]
fn float_field_different_seeds_different_output() {
    let field1 = generate_float_field(10, 10, 10, 0.0, 1);
    let field2 = generate_float_field(10, 10, 10, 0.0, 42);
    assert_ne!(field1, field2);
}

#[test]
fn float_field_trivial_first_element_is_offset() {
    let trivial = generate_float_field(1, 1, 1, 123.456, 1);
    assert_eq!(trivial[0], 123.456f32);
}

#[test]
fn double_field_trivial_first_element_is_offset() {
    let trivial = generate_double_field(1, 1, 1, 123.456, 1);
    assert_eq!(trivial[0], 123.456f64);
}

#[test]
fn float_field_standard_dimensions() {
    let field = generate_float_field(65, 64, 63, 0.0, 1);
    assert_eq!(field.len(), 65 * 64 * 63);
    assert_eq!(field.len(), 262080);
}

#[test]
fn double_field_standard_dimensions() {
    let field = generate_double_field(65, 64, 63, 0.0, 1);
    assert_eq!(field.len(), 65 * 64 * 63);
    assert_eq!(field.len(), 262080);
}

#[test]
fn float_field_has_reasonable_range() {
    let field = generate_float_field(10, 10, 10, 0.0, 1);
    assert!(field.iter().all(|v| v.is_finite()));
    let min = field.iter().cloned().reduce(f32::min).unwrap();
    let max = field.iter().cloned().reduce(f32::max).unwrap();
    assert!(max > min, "Field should have variation");
}
