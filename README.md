# rcf-fc-interners

This crate implements a concurrent read-contention-free (RCF) fixed-capacity (FC) `HashMap`-based interner.

> **NB**: *Contention-free* is not the same as *lock-free*.
> 
> All contention-free code is lock-free, but not vice versa.
>
> Code can be lock-free yet not contention-free — for example, if it relies on `compare_exchange` operations on atomics,
> or any other operations with a memory ordering stronger than `Relaxed`.

## Features & Advantages
1. Stores key–value pairs, similar to an associative map.  
   For each unique key, insertion returns an interned handle — a unique, monotonically incremented `usize` ID.
2. *Contention-free* for reading key–value pairs (as well as keys or values separately) by their assigned IDs.
3. *Fixed capacity* — all memory is allocated upfront when the first key is inserted.

## Limitations
1. Does *not* perform memory reclamation until the interner is dropped.
2. Writes are *not* optimized — they incur overhead due to the internal `RwLock`.
3. Re-insertion of an existing key is *not* supported.
4. Deletion is *not* supported.

## Usage Patterns
1. The total number of key–value pairs is known in advance and strictly bounded from above.
2. The total memory required to store all key–value pairs can be pre-allocated with sufficient margin.
3. Interning operations are performed much less frequently than reads.
