extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;

/// Circular buffer for the 3D wavefront predictor.
pub struct Front<T: Copy + Default> {
    zero: T,
    dx: i32,
    dy: i32,
    dz: i32,
    mask: i32,
    index: i32,
    buffer: Vec<T>,
}

impl<T: Copy + Default> Front<T> {
    /// Creates a new Front buffer for the given dimensions.
    pub fn new(nx: i32, ny: i32, zero: T) -> Self {
        let dx = 1;
        let dy = nx + 1;
        let dz = dy * (ny + 1);
        let mask = compute_mask(dx + dy + dz);
        Self {
            zero,
            dx,
            dy,
            dz,
            mask,
            index: 0,
            buffer: vec![T::default(); (mask + 1) as usize],
        }
    }

    /// Fetches a neighbor relative to the current sample position.
    #[inline]
    pub fn get(&self, x: i32, y: i32, z: i32) -> T {
        let idx = (self.index - self.dx * x - self.dy * y - self.dz * z) & self.mask;
        self.buffer[idx as usize]
    }

    /// Adds a sample to the front.
    #[inline]
    pub fn push(&mut self, value: T) {
        self.buffer[(self.index & self.mask) as usize] = value;
        self.index += 1;
    }

    /// Adds n copies of a value to the front.
    pub fn push_n(&mut self, value: T, count: i32) {
        for _ in 0..count {
            self.buffer[(self.index & self.mask) as usize] = value;
            self.index += 1;
        }
    }

    /// Advances the front, filling with zeros.
    pub fn advance(&mut self, x: i32, y: i32, z: i32) {
        self.push_n(self.zero, self.dx * x + self.dy * y + self.dz * z);
    }
}

/// Computes the smallest power-of-2-minus-1 mask that is >= n-1.
fn compute_mask(mut n: i32) -> i32 {
    n -= 1;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_to_zero() {
        let f = Front::<u32>::new(4, 4, 0);
        assert_eq!(f.get(0, 0, 0), 0);
        assert_eq!(f.get(1, 0, 0), 0);
        assert_eq!(f.get(0, 1, 0), 0);
        assert_eq!(f.get(0, 0, 1), 0);
    }

    #[test]
    fn push_and_retrieve() {
        let mut f = Front::<u32>::new(4, 4, 0);
        f.push(42);
        // After push, index advanced by 1, so get(1,0,0) should be 42
        // since dx=1 and the value was at old index
        assert_eq!(f.get(1, 0, 0), 42);
    }

    #[test]
    fn advance_fills_zeros() {
        let mut f = Front::<u32>::new(4, 4, 0);
        f.push(99);
        f.advance(1, 0, 0); // advance by dx=1 slot, filling with zero
        assert_eq!(f.get(1, 0, 0), 0);
        // The 99 should now be at distance 2 in the x direction
        assert_eq!(f.get(2, 0, 0), 99);
    }

    #[test]
    fn circular_buffer_wraps() {
        let mut f = Front::<u32>::new(2, 2, 0);
        let buf_size = (f.mask + 1) as u32;
        // Fill entire buffer and beyond
        for i in 0..buf_size * 2 {
            f.push(i);
        }
        // Should still work (wrapped around)
        let last = buf_size * 2 - 1;
        assert_eq!(f.get(1, 0, 0), last);
    }
}
