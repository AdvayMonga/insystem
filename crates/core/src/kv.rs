//! Paged KV block allocator with prefix caching.
//!
//! Blocks hold `block_size` tokens. A full block is identified by a hash chained over its parent
//! block's hash and its own tokens, so equal hashes mean equal prefixes. A block whose reference
//! count drops to zero goes to the tail of an LRU free list but keeps its hash: a later request
//! with the same prefix takes it back off the list. Allocation pops the head of the list (the
//! least recently freed block) and drops its hash.

use std::collections::hash_map::DefaultHasher;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

const NIL: u32 = u32::MAX;

pub fn chain_hash(parent: u64, tokens: &[u32]) -> u64 {
    let mut h = DefaultHasher::new();
    parent.hash(&mut h);
    tokens.hash(&mut h);
    h.finish()
}

#[derive(Clone, Copy)]
struct Block {
    ref_cnt: u32,
    hash: Option<u64>,
    prev: u32,
    next: u32,
    in_free: bool,
}

pub struct BlockManager {
    pub block_size: usize,
    blocks: Vec<Block>,
    head: u32,
    tail: u32,
    num_free: usize,
    cached: HashMap<u64, u32>,
    pub prefix_caching: bool,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize, prefix_caching: bool) -> Self {
        let mut m = Self {
            block_size,
            blocks: vec![Block { ref_cnt: 0, hash: None, prev: NIL, next: NIL, in_free: false }; num_blocks],
            head: NIL,
            tail: NIL,
            num_free: 0,
            cached: HashMap::new(),
            prefix_caching,
        };
        for b in 0..num_blocks as u32 {
            m.push_tail(b);
        }
        m
    }

    pub fn num_blocks(&self) -> usize {
        self.blocks.len()
    }
    pub fn num_free(&self) -> usize {
        self.num_free
    }
    pub fn num_cached(&self) -> usize {
        self.cached.len()
    }

    fn push_tail(&mut self, b: u32) {
        let blk = &mut self.blocks[b as usize];
        debug_assert!(!blk.in_free);
        blk.in_free = true;
        blk.prev = self.tail;
        blk.next = NIL;
        if self.tail != NIL {
            self.blocks[self.tail as usize].next = b;
        } else {
            self.head = b;
        }
        self.tail = b;
        self.num_free += 1;
    }

    fn unlink(&mut self, b: u32) {
        let Block { prev, next, in_free, .. } = self.blocks[b as usize];
        debug_assert!(in_free);
        if prev != NIL {
            self.blocks[prev as usize].next = next;
        } else {
            self.head = next;
        }
        if next != NIL {
            self.blocks[next as usize].prev = prev;
        } else {
            self.tail = prev;
        }
        let blk = &mut self.blocks[b as usize];
        blk.in_free = false;
        blk.prev = NIL;
        blk.next = NIL;
        self.num_free -= 1;
    }

    /// Takes a reference on each cached full block matching a prefix of `tokens`, stopping so that
    /// at least one token is left to compute. Returns the blocks and their hashes.
    pub fn acquire_prefix(&mut self, tokens: &[u32]) -> (Vec<u32>, Vec<u64>) {
        let mut blocks = Vec::new();
        let mut hashes = Vec::new();
        if !self.prefix_caching || tokens.len() < 2 {
            return (blocks, hashes);
        }
        let bs = self.block_size;
        let max_full = (tokens.len() - 1) / bs;
        let mut parent = 0u64;
        for i in 0..max_full {
            let h = chain_hash(parent, &tokens[i * bs..(i + 1) * bs]);
            match self.cached.get(&h) {
                Some(&b) => {
                    if self.blocks[b as usize].in_free {
                        self.unlink(b);
                    }
                    self.blocks[b as usize].ref_cnt += 1;
                    blocks.push(b);
                    hashes.push(h);
                    parent = h;
                }
                None => break,
            }
        }
        (blocks, hashes)
    }

    pub fn can_allocate(&self, n: usize) -> bool {
        self.num_free >= n
    }

    pub fn allocate(&mut self, n: usize) -> Option<Vec<u32>> {
        if self.num_free < n {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            let b = self.head;
            self.unlink(b);
            let blk = &mut self.blocks[b as usize];
            if let Some(h) = blk.hash.take() {
                if self.cached.get(&h) == Some(&b) {
                    self.cached.remove(&h);
                }
            }
            blk.ref_cnt = 1;
            out.push(b);
        }
        Some(out)
    }

    /// Drops one reference on each block. Blocks are released in reverse so a sequence's tail
    /// blocks are evicted before its shared head blocks.
    pub fn release(&mut self, blocks: &[u32]) {
        for &b in blocks.iter().rev() {
            let blk = &mut self.blocks[b as usize];
            debug_assert!(blk.ref_cnt > 0);
            blk.ref_cnt -= 1;
            if blk.ref_cnt == 0 {
                if blk.hash.is_none() {
                    // Uncached blocks go to the head: they are reused first.
                    self.push_head(b);
                } else {
                    self.push_tail(b);
                }
            }
        }
    }

    fn push_head(&mut self, b: u32) {
        let old = self.head;
        let blk = &mut self.blocks[b as usize];
        blk.in_free = true;
        blk.prev = NIL;
        blk.next = old;
        if old != NIL {
            self.blocks[old as usize].prev = b;
        } else {
            self.tail = b;
        }
        self.head = b;
        self.num_free += 1;
    }

    /// Records the hash of a block that has just become full and computed.
    pub fn register(&mut self, block: u32, hash: u64) {
        if !self.prefix_caching {
            return;
        }
        if self.cached.contains_key(&hash) {
            return;
        }
        self.blocks[block as usize].hash = Some(hash);
        self.cached.insert(hash, block);
    }

    /// Drops every cached hash (the blocks themselves stay where they are).
    pub fn reset_prefix_cache(&mut self) {
        for (_, b) in self.cached.drain() {
            self.blocks[b as usize].hash = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_hit_after_release() {
        let mut m = BlockManager::new(8, 4, true);
        let toks: Vec<u32> = (0..10).collect();
        let a = m.allocate(3).unwrap();
        let mut parent = 0;
        for i in 0..2 {
            let h = chain_hash(parent, &toks[i * 4..(i + 1) * 4]);
            m.register(a[i], h);
            parent = h;
        }
        m.release(&a);
        assert_eq!(m.num_free(), 8);
        let (hit, _) = m.acquire_prefix(&toks);
        assert_eq!(hit, vec![a[0], a[1]]);
        assert_eq!(m.num_free(), 6);
        // A prompt of exactly 8 tokens reuses only the first block, leaving a token to compute.
        let (hit2, _) = m.acquire_prefix(&toks[..8]);
        assert_eq!(hit2, vec![a[0]]);
    }

    #[test]
    fn allocation_evicts_lru_cached_blocks() {
        let mut m = BlockManager::new(2, 2, true);
        let a = m.allocate(2).unwrap();
        m.register(a[0], 11);
        m.register(a[1], 22);
        m.release(&a);
        let b = m.allocate(2).unwrap();
        assert_eq!(b.len(), 2);
        assert_eq!(m.num_cached(), 0);
    }
}
