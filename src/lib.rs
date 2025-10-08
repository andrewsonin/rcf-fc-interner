//! This crate implements a concurrent read-contention-free (RCF) fixed-capacity
//! (FC) `HashMap`-based interner.
//!
//! > **NB**: *Contention-free* is not the same as *lock-free*.
//! >
//! > All contention-free code is lock-free, but not vice versa.
//! >
//! > Code can be lock-free yet not contention-free — for example, if it relies
//! > on `compare_exchange` operations on atomics,
//! > or any other operations with a memory ordering stronger than `Relaxed`.
//!
//! ## Features & Advantages
//! 1. Stores key–value pairs, similar to an associative map. For each unique
//!    key, insertion returns an interned handle — a unique, monotonically
//!    incremented `usize` ID.
//! 2. *Contention-free* for reading key–value pairs (as well as keys or values
//!    separately) by their assigned IDs.
//! 3. *Fixed capacity* — all memory is allocated upfront when the first key is
//!    inserted.
//!
//! ## Limitations
//! 1. Does *not* perform memory reclamation until the interner is dropped.
//! 2. Writes are *not* optimized — they incur overhead due to the internal
//!    `RwLock`.
//! 3. Re-insertion of an existing key is *not* supported.
//! 4. Deletion is *not* supported.
//!
//! ## Usage Patterns
//! 1. The total number of key–value pairs is known in advance and strictly
//!    bounded from above.
//! 2. The total memory required to store all key–value pairs can be
//!    pre-allocated with sufficient margin.
//! 3. Interning operations are performed much less frequently than reads.

use core::{
    alloc::Layout,
    any::type_name,
    cell::UnsafeCell,
    fmt::{Debug, Formatter},
    hash::{BuildHasher, Hash},
    marker::PhantomData,
    mem::{MaybeUninit, needs_drop},
    num::NonZeroUsize,
    ptr::{NonNull, drop_in_place},
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::atomic::{AtomicUsize, Ordering},
};
use parking_lot::{RwLock, RwLockWriteGuard};
use std::{
    alloc::{alloc, dealloc, handle_alloc_error, realloc},
    collections::{
        HashMap,
        hash_map::{Entry, VacantEntry},
    },
    hash::RandomState,
    ptr::copy_nonoverlapping,
};

/// Read-contention-free (RCF) fixed-capacity (FC)
/// [`HashMap`]-based interner.
pub struct RCFFCHashMap<K, V, S = RandomState> {
    /// If initialized, it points to a single allocated object of a type
    /// `[MaybeUninit<K>; cap]`.
    ///
    /// The current synchronized non-zero value of `len` means
    /// that the first `len` objects inside it are initialized and can no longer
    /// be modified.
    keys: UnsafeCell<NonNull<K>>,
    /// If initialized, it points to a single allocated object of a type
    /// `[MaybeUninit<V>; cap]`.
    ///
    /// The current synchronized non-zero value of `len` means
    /// that the first `len` objects inside it are initialized and can no longer
    /// be modified via shared reference.
    values: UnsafeCell<NonNull<V>>,
    /// Contains key to offset mapping.
    ///
    /// The current synchronized non-zero value of `len` means that the
    /// `MaybeUninit` is initialized.
    key_to_offset: RwLock<MaybeUninit<HashMap<K, usize, S>>>,
    /// Current length of the structure.
    /// * `0` — means it may be not initialized.
    /// * Any synchronized non-zero value means that it's fully initialized:
    ///     * `MaybeUninit` is init.
    ///     * `NonNull` values can't longer be modified, aren't dangling and
    ///       point to a single, owned allocated object with `cap` length.
    len: AtomicUsize,
    /// Maximum capacity of the interner.
    cap: NonZeroUsize,
    /// Drop-check fantom. It tells the compiler that this structure owns
    /// instances of types `K`, `V` and `S`.
    _dropck_phantom: PhantomData<(K, V, S)>,
}

unsafe impl<K, V, S> Sync for RCFFCHashMap<K, V, S>
where
    K: Send + Sync, // `RwLock: Sync` requirement
    V: Sync,
    S: Send + Sync, // `RwLock: Sync` requirement
{
}

unsafe impl<K, V, S> Send for RCFFCHashMap<K, V, S>
where
    K: Send,
    V: Send,
    S: Send,
{
}

impl<K, V, S> RCFFCHashMap<K, V, S> {
    /// Creates a new instance of `Self`.
    ///
    /// # Arguments
    /// * `capacity` — maximum capacity of the interner.
    #[inline]
    #[must_use]
    pub const fn new(capacity: NonZeroUsize) -> Self {
        Self {
            keys: UnsafeCell::new(NonNull::dangling()),
            values: UnsafeCell::new(NonNull::dangling()),
            key_to_offset: RwLock::new(MaybeUninit::uninit()),
            len: AtomicUsize::new(0),
            cap: capacity,
            _dropck_phantom: PhantomData,
        }
    }

    /// Returns the current **synchronized** length of the interner.
    ///
    /// That is, all items which offsets are less than this length are visible
    /// to the current thread.
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }

    /// Returns whether the interner isn't initialized (i.e., its synchronized
    /// length isn't equal to zero).
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns whether the interner isn't initialized (i.e., its synchronized
    /// length isn't equal to zero), like [`is_empty`](Self::is_empty) does, but
    /// without the risk of contention due to unique access.
    #[inline]
    #[must_use]
    pub fn is_empty_exclusive(&mut self) -> bool {
        self.len_exclusive() == 0
    }

    /// Returns the current interner capacity.
    #[inline]
    #[must_use]
    pub fn capacity(&self) -> NonZeroUsize {
        self.cap
    }

    /// Returns the current **synchronized** length of the interner, like
    /// [`len`](Self::len) does, but without the risk of contention due to
    /// unique access.
    #[inline]
    #[must_use]
    pub fn len_exclusive(&mut self) -> usize {
        *self.len.get_mut()
    }

    /// Returns whether the interner is initialized (i.e., its synchronized
    /// length isn't equal to zero).
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn is_initialized(&self) -> bool {
        !self.is_empty()
    }

    /// Returns whether the interner is initialized, like
    /// [`len`](Self::is_initialized) does, but without the risk of
    /// contention due to unique access.
    #[inline]
    #[must_use]
    pub fn is_initialized_exclusive(&mut self) -> bool {
        self.len_exclusive() != 0
    }

    /// Returns an immutable reference to the initialized part of the RCF `key`
    /// storage.
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn keys(&self) -> &[K] {
        let len = NonZeroUsize::new(self.len());
        unsafe {
            if let Some(len) = len {
                self.keys_unchecked(len)
            } else {
                // If len is zero, interner can be in the process of initialization, so
                // don't use `self.keys` here to avoid potential data race.
                from_raw_parts(NonNull::<K>::dangling().as_ptr(), 0)
            }
        }
    }

    /// Returns an immutable reference to the initialized part of the RCF `key`
    /// storage, like [`keys`](Self::keys) does, but without the risk of
    /// contention due to unique access.
    #[inline]
    #[must_use]
    pub fn keys_exclusive(&mut self) -> &[K] {
        self.keys_mut()
    }

    /// Returns a mutable reference to the initialized part of the RCF `key`
    /// storage.
    ///
    /// This function is private and is called inside `Drop` only,
    /// since keys cannot be mutated at runtime without modifying the
    /// corresponding entry in the `key_to_offset` map.
    #[inline]
    #[must_use]
    fn keys_mut(&mut self) -> &mut [K] {
        let len = *self.len.get_mut();
        let ptr = self.keys.get_mut().as_ptr();
        unsafe { from_raw_parts_mut(ptr, len) }
    }

    /// Returns an immutable reference to the initialized part of the RCF
    /// `value` storage.
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn values(&self) -> &[V] {
        let len = NonZeroUsize::new(self.len());
        unsafe {
            if let Some(len) = len {
                self.values_unchecked(len)
            } else {
                // If len is zero, interner can be in the process of initialization, so
                // don't use `self.values` here to avoid potential data race.
                from_raw_parts(NonNull::<V>::dangling().as_ptr(), 0)
            }
        }
    }

    /// Returns a mutable reference to the initialized part of the RCF `value`
    /// storage.
    #[inline]
    #[must_use]
    pub fn values_mut(&mut self) -> &mut [V] {
        let len = *self.len.get_mut();
        unsafe { self.values_unchecked_mut(len) }
    }

    /// Returns an unchecked immutable reference to the initialized part of the
    /// RCF `key` storage.
    ///
    /// # Arguments
    /// * `len` — resulting length.
    ///
    /// # Safety
    /// The item which corresponds to offset `len - 1` is initialized,
    /// and it's visible to the current thread.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn keys_unchecked(&self, len: NonZeroUsize) -> &[K] {
        let ptr = self.keys.get().read().as_ptr();
        from_raw_parts(ptr, len.get())
    }

    /// Returns an unchecked immutable reference to the initialized part of the
    /// RCF `value` storage.
    ///
    /// # Arguments
    /// * `len` — resulting length.
    ///
    /// # Safety
    /// The item which corresponds to offset `len - 1` is initialized,
    /// and it's visible to the current thread.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn values_unchecked(&self, len: NonZeroUsize) -> &[V] {
        let ptr = self.values.get().read().as_ptr();
        from_raw_parts(ptr, len.get())
    }

    /// Returns an unchecked mutable reference to the initialized part of the
    /// RCF `value` storage.
    ///
    /// # Arguments
    /// * `len` — resulting length.
    ///
    /// # Safety
    /// The item which corresponds to offset `len - 1` is initialized, or `len`
    /// is `0`.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn values_unchecked_mut(&mut self, len: usize) -> &mut [V] {
        let ptr = self.values.get_mut().as_ptr();
        from_raw_parts_mut(ptr, len)
    }

    /// Returns an immutable reference to `key` corresponding to the
    /// provided offset if it's already initialized.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn key(&self, offset: usize) -> Option<&K> {
        if offset < self.len() {
            // SAFETY: the item which corresponds to `offset` is initialized, and it's
            // visible to the current thread.
            Some(unsafe { self.key_unchecked(offset) })
        } else {
            None
        }
    }

    /// Returns an unchecked immutable reference to `key` corresponding to the
    /// provided offset.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    ///
    /// # Safety
    /// The item which corresponds to `offset` is initialized,
    /// and it's visible to the current thread.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn key_unchecked(&self, offset: usize) -> &K {
        self.keys
            .get()
            .read()
            .as_ptr()
            .add(offset)
            .as_ref()
            .unwrap_unchecked()
    }

    /// Returns an immutable reference to `value` corresponding to the
    /// provided offset if it's already initialized.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    ///
    /// # Performance
    /// This method can cause contention.
    #[inline]
    #[must_use]
    pub fn value(&self, offset: usize) -> Option<&V> {
        if offset < self.len() {
            // SAFETY: the item which corresponds to `offset` is initialized, and it's
            // visible to the current thread.
            Some(unsafe { self.value_unchecked(offset) })
        } else {
            None
        }
    }

    /// Returns a mutable reference to `value` corresponding to the
    /// provided offset if it's already initialized.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    #[inline]
    #[must_use]
    pub fn value_mut(&mut self, offset: usize) -> Option<&mut V> {
        if offset < self.len_exclusive() {
            // SAFETY: the item which corresponds to `offset` is initialized.
            Some(unsafe { self.value_unchecked_mut(offset) })
        } else {
            None
        }
    }

    /// Returns an unchecked immutable reference to `value` corresponding to the
    /// provided offset.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    ///
    /// # Safety
    /// The item which corresponds to `offset` is initialized,
    /// and it's visible to the current thread.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn value_unchecked(&self, offset: usize) -> &V {
        self.values
            .get()
            .read()
            .as_ptr()
            .add(offset)
            .as_ref()
            .unwrap_unchecked()
    }

    /// Returns an unchecked mutable reference to `value` corresponding to the
    /// provided offset.
    ///
    /// # Arguments
    /// * `offset` — interning offset.
    ///
    /// # Safety
    /// The item which corresponds to `offset` is initialized.
    #[inline]
    #[must_use]
    #[allow(unsafe_op_in_unsafe_fn)]
    pub unsafe fn value_unchecked_mut(&mut self, offset: usize) -> &mut V {
        self.values
            .get_mut()
            .as_ptr()
            .add(offset)
            .as_mut()
            .unwrap_unchecked()
    }
}

/// [`RCFFCHashMap::intern_ref`] result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InternRefResult<K, V> {
    /// Successful internalization offset (key-value pair internalization ID).
    Interned(usize),
    /// Indicates that the key already exists.
    KeyAlreadyExists {
        /// Existing key-value pair internalization ID.
        offset: usize,
        /// Value that wasn't inserted.
        value: V,
    },
    /// Indicates that the capacity overflowed.
    CapacityOverflow {
        /// Key that wasn't inserted.
        key: K,
        /// Value that wasn't inserted.
        value: V,
    },
}

impl<K, V, S> RCFFCHashMap<K, V, S>
where
    K: Clone + Eq + Hash,
    S: Default + BuildHasher,
{
    /// Reserves capacity for `additional` more elements to be inserted
    /// in the given `RCFFCHashMap<K, V, S>`.
    ///
    /// # Arguments
    /// * `additional` — additional capacity of the interner.
    ///
    /// # Panics
    /// * `capacity` overflow happens.
    /// * Reallocation fails.
    #[inline]
    #[track_caller]
    pub fn reserve(&mut self, additional: usize) {
        if !self.is_initialized_exclusive() {
            self.cap = self.cap.checked_add(additional).unwrap_or_else(|| {
                panic!(
                    "{Self}::reserve: capacity `usize` overflow: {cap} + {additional}",
                    Self = type_name::<Self>(),
                    cap = self.cap,
                )
            });
            return;
        }

        let Self {
            keys,
            values,
            key_to_offset,
            len,
            cap,
            _dropck_phantom,
        } = self;

        // SAFETY: see the `Safety` section of a method.
        let realloc_result = unsafe {
            Self::realloc(
                *keys.get_mut(),
                *values.get_mut(),
                *cap,
                *len.get_mut(),
                additional,
            )
        };
        let (new_cap, keys_ptr, values_ptr) = match realloc_result {
            ReallocResult::Success {
                keys,
                values,
                new_cap,
            } => (new_cap, keys, values),
            ReallocResult::KeysFailed { keys_layout } => {
                handle_alloc_error(keys_layout);
            }
            ReallocResult::ValuesFailed {
                keys: keys_ptr,
                values_layout,
            } => {
                // Update `keys` with a valid reallocated pointer to avoid getting UB during a
                // possible drop.
                *keys.get_mut() = keys_ptr;
                handle_alloc_error(values_layout);
            }
        };
        *keys.get_mut() = keys_ptr;
        *values.get_mut() = values_ptr;
        *cap = new_cap;
        // SAFETY: the interner is fully initialized.
        let key_to_offset = unsafe { key_to_offset.get_mut().assume_init_mut() };
        key_to_offset.reserve(additional);
    }

    /// Concurrently internalizes `key` with `value`.
    ///
    /// # Arguments
    /// * `key` — key.
    /// * `value` — value.
    ///
    /// # Performance
    /// * This method can be blocking to other threads.
    /// * This method causes contention.
    #[inline]
    pub fn intern_ref(&self, key: K, value: V) -> InternRefResult<K, V> {
        match self.intern_ref_with(key, || value) {
            InternRefResult::Interned(id) => InternRefResult::Interned(id),
            InternRefResult::KeyAlreadyExists { offset, value: f } => {
                InternRefResult::KeyAlreadyExists { offset, value: f() }
            }
            InternRefResult::CapacityOverflow { key, value: f } => {
                InternRefResult::CapacityOverflow { key, value: f() }
            }
        }
    }

    /// Concurrently internalizes `key` with the corresponding value lazily
    /// provided by `f`.
    ///
    /// # Arguments
    /// * `key` — key.
    /// * `f` — value lazy initializer.
    ///
    /// # Performance
    /// * This method can be blocking to other threads.
    /// * This method causes contention.
    #[inline]
    pub fn intern_ref_with<F: FnOnce() -> V>(&self, key: K, f: F) -> InternRefResult<K, F> {
        let mut guard = self.key_to_offset.read();

        let map = 'map: {
            // `Relaxed` ordering is sufficient here, since it happens inside a critical
            // section of the `RwLock` and no concurrent modifications of this variable take
            // place outside of this section.
            if self.len.load(Ordering::Relaxed) != 0 {
                // SAFETY: the interner is initialized.
                unsafe { guard.assume_init_ref() }
            } else {
                drop(guard);

                let w_guard = self.key_to_offset.write();
                // `Relaxed` ordering is sufficient here, since it happens inside a critical
                // section of the `RwLock` and no concurrent modifications of this variable take
                // place outside of this section.
                if self.len.load(Ordering::Relaxed) != 0 {
                    guard = RwLockWriteGuard::downgrade(w_guard);
                    // SAFETY: the interner is initialized.
                    let map = unsafe { guard.assume_init_ref() };
                    break 'map map;
                }

                // SAFETY: there are no shared accesses to `keys` and `values`.
                unsafe {
                    Self::init(
                        key,
                        f(),
                        w_guard,
                        &self.keys,
                        &self.values,
                        &self.len,
                        self.cap,
                    );
                }

                return InternRefResult::Interned(0);
            }
        };

        if let Some(&offset) = map.get(&key) {
            return InternRefResult::KeyAlreadyExists { offset, value: f };
        }
        drop(guard);

        let mut w_guard = self.key_to_offset.write();
        // SAFETY: the interner is initialized.
        let map = unsafe { w_guard.assume_init_mut() };

        // `Relaxed` ordering is sufficient here, since it happens inside a critical
        // section of the `RwLock` and no concurrent modifications of this variable take
        // place outside of this section.
        if self.len.load(Ordering::Relaxed) == self.cap.get() {
            return InternRefResult::CapacityOverflow { key, value: f };
        }

        match map.entry(key) {
            Entry::Vacant(entry) => {
                // SAFETY:
                // * `entry` is guarded by the `RwLockWriteGuard` protecting the `key_to_offset`
                //   field.
                // * This guard guarantees that there are no shared accesses to `keys` and
                //   `values`.
                // * No `cap` overflow.
                let offset =
                    unsafe { Self::write_vacant(entry, f, &self.keys, &self.values, &self.len) };
                InternRefResult::Interned(offset)
            }
            Entry::Occupied(entry) => InternRefResult::KeyAlreadyExists {
                offset: *entry.get(),
                value: f,
            },
        }
    }

    /// # Safety:
    /// * `w_guard` guarantees that there are no shared accesses to `keys` and
    ///   `values`.
    #[cold]
    unsafe fn init(
        key: K,
        value: V,
        mut w_guard: RwLockWriteGuard<'_, MaybeUninit<HashMap<K, usize, S>>>,
        keys: &UnsafeCell<NonNull<K>>,
        values: &UnsafeCell<NonNull<V>>,
        len: &AtomicUsize,
        cap: NonZeroUsize,
    ) {
        let (keys_ptr, values_ptr) = Self::alloc(cap);

        // SAFETY: there are no shared accesses to `keys` and `values`.
        unsafe {
            *keys.get().as_mut().unwrap_unchecked() = keys_ptr;
            *values.get().as_mut().unwrap_unchecked() = values_ptr;

            keys_ptr.write(key.clone());
            values_ptr.write(value);
        }

        let map = w_guard.write(HashMap::with_capacity_and_hasher(cap.get(), S::default()));
        // Here the initialization flag is stored.
        //
        // It has a `Release` ordering because upstream changes should be visible to
        // other threads if they see a synchronized non-zero length,
        // which is a guarantee that the structure is fully initialized.
        //
        // It also guarantees that the first
        // elements in the allocated objects of `keys` and `values` are
        // initialized and can no longer be modified.
        //
        // We also don't need to synchronize the insertion below -
        // only the initialization of `MaybeUninit`, which is used in the destructor.
        //
        // The insertion below does not need to be synchronized using this atomic
        // variable, since `RwLock` does that.
        len.store(1, Ordering::Release);

        map.insert(key, 0);
    }

    /// # Safety
    /// * `entry` is guarded by the [`RwLockWriteGuard`] protecting the
    ///   `key_to_offset` field.
    /// * This guard guarantees that there are no shared accesses to `keys` and
    ///   `values`.
    /// * No `cap` overflow.
    #[inline]
    unsafe fn write_vacant<F: FnOnce() -> V>(
        entry: VacantEntry<'_, K, usize>,
        f: F,
        keys: &UnsafeCell<NonNull<K>>,
        values: &UnsafeCell<NonNull<V>>,
        len: &AtomicUsize,
    ) -> usize {
        // `Relaxed` ordering is sufficient here, since it happens inside a critical
        // section of the `RwLock` and no concurrent modifications of this variable take
        // place outside of this section.
        let offset = len.load(Ordering::Relaxed);

        // SAFETY: there are no shared accesses to the vacant slots.
        unsafe {
            let key_slot = keys.get().read().as_ptr().add(offset);
            let value_slot = values.get().read().as_ptr().add(offset);

            key_slot.write(entry.key().clone());
            value_slot.write(f());
        }

        // It has a `Release` ordering because upstream changes should be visible to
        // other threads if they see a synchronized non-zero length,
        // which is a guarantee that the structure is fully initialized.
        //
        // It also guarantees that the first `len`
        // elements in the allocated objects of `keys` and `values` are
        // initialized and can no longer be modified.
        //
        // We also don't need to synchronize the insertion below -
        // only the initialization of `MaybeUninit`, which is used in the destructor.
        //
        // The insertion below does not need to be synchronized using this atomic
        // variable, since `RwLock` does that.
        len.store(offset + 1, Ordering::Release);

        entry.insert(offset);

        offset
    }

    #[inline]
    #[track_caller]
    fn alloc(cap: NonZeroUsize) -> (NonNull<K>, NonNull<V>) {
        let layout_keys = Layout::array::<K>(cap.get()).unwrap_or_else(|err| {
            panic!(
                "{Self}::alloc: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<K>()
            )
        });
        let layout_values = Layout::array::<V>(cap.get()).unwrap_or_else(|err| {
            panic!(
                "{Self}::alloc: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<V>()
            )
        });

        // SAFETY: layout size is non-zero.
        let keys = unsafe { alloc(layout_keys) };
        let Some(keys) = NonNull::new(keys.cast::<K>()) else {
            handle_alloc_error(layout_keys)
        };

        // SAFETY: layout size is non-zero.
        let values = unsafe { alloc(layout_values) };
        let Some(values) = NonNull::new(values.cast::<V>()) else {
            // SAFETY:
            // * Keys are allocated.
            // * Layout is same.
            unsafe {
                dealloc(keys.as_ptr().cast::<u8>(), layout_keys);
            }
            handle_alloc_error(layout_values)
        };
        (keys, values)
    }

    /// # Safety
    /// * `keys` point to the allocated object of a type `[MaybeUninit<K>;
    ///   cap]`.
    /// * `values` point to the allocated object of a type `[MaybeUninit<V>;
    ///   cap]`.
    /// * These allocated objects are allocated using the global allocator.
    #[inline]
    #[track_caller]
    unsafe fn realloc(
        keys: NonNull<K>,
        values: NonNull<V>,
        cap: NonZeroUsize,
        len: usize,
        additional: usize,
    ) -> ReallocResult<K, V> {
        let new_cap = cap.checked_add(additional).unwrap_or_else(|| {
            panic!(
                "{Self}::realloc: capacity `usize` overflow: {cap} + {additional}",
                Self = type_name::<Self>(),
            )
        });

        let layout_keys = Layout::array::<K>(cap.get()).unwrap_or_else(|err| {
            unreachable!(
                "{Self}::realloc: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<K>(),
                err = err,
            )
        });
        let layout_values = Layout::array::<V>(cap.get()).unwrap_or_else(|err| {
            unreachable!(
                "{Self}::realloc: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<V>(),
                err = err,
            )
        });

        // SAFETY: see the `Safety` section of a method.
        let new_layout_keys = Layout::array::<K>(new_cap.get()).unwrap_or_else(|err| {
            unreachable!(
                "{Self}::realloc: \
                 can't create array layout for type `{t}` of {new_cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<K>(),
                err = err,
            )
        });
        let new_keys = unsafe { alloc(new_layout_keys) };
        let Some(new_keys) = NonNull::new(new_keys.cast::<K>()) else {
            return ReallocResult::KeysFailed {
                keys_layout: new_layout_keys,
            };
        };

        // SAFETY: see the `Safety` section of a method.
        let new_layout_values = Layout::array::<V>(new_cap.get()).unwrap_or_else(|err| {
            unreachable!(
                "{Self}::realloc: \
                 can't create array layout for type `{t}` of {new_cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<V>(),
                err = err,
            )
        });
        let new_values = unsafe {
            realloc(
                values.as_ptr().cast::<u8>(),
                layout_values,
                new_layout_values.size(),
            )
        };
        let Some(values) = NonNull::new(new_values.cast::<V>()) else {
            unsafe {
                dealloc(new_keys.as_ptr().cast::<u8>(), new_layout_keys);
            }
            return ReallocResult::ValuesFailed {
                keys,
                values_layout: new_layout_values,
            };
        };

        unsafe {
            copy_nonoverlapping(keys.as_ptr(), new_keys.as_ptr(), len);
            dealloc(keys.as_ptr().cast::<u8>(), layout_keys);
        }
        let keys = new_keys;

        ReallocResult::Success {
            new_cap,
            keys,
            values,
        }
    }
}

impl<K, V, S> Debug for RCFFCHashMap<K, V, S>
where
    K: Debug,
    V: Debug,
{
    #[inline]
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut f = f.debug_struct(type_name::<Self>());
        let guard = self.key_to_offset.read();
        let len = self.len.load(Ordering::Relaxed);
        if len != 0 {
            f.field("key_to_offset", &*guard);
            drop(guard);
            // SAFETY:
            // Items with offset less than `len` are initialized,
            // and this fact is visible to the current thread.
            unsafe {
                let len = NonZeroUsize::new_unchecked(len);
                f.field("keys", &self.keys_unchecked(len));
                f.field("values", &self.values_unchecked(len));
            }
        } else {
            drop(guard);
        }
        f.field("len", &len);
        f.field("cap", &self.cap);
        f.finish()
    }
}

impl<K, V, S> Drop for RCFFCHashMap<K, V, S> {
    #[inline]
    fn drop(&mut self) {
        if !self.is_initialized_exclusive() {
            return;
        }

        let cap = self.cap.get();

        if needs_drop::<K>() {
            for key in self.keys_mut() {
                // SAFETY: `key` is initialized.
                unsafe { drop_in_place(key) }
            }
        }

        let layout = Layout::array::<K>(cap).unwrap_or_else(|err| {
            unreachable!(
                "<{Self} as Drop>::drop: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<K>(),
                cap = cap,
                err = err,
            )
        });
        // SAFETY: `keys` points to the object allocated with `layout`.
        unsafe { dealloc(self.keys.get_mut().as_ptr().cast::<u8>(), layout) }

        if needs_drop::<V>() {
            for value in self.values_mut() {
                // SAFETY: `value` is initialized.
                unsafe { drop_in_place(value) }
            }
        }

        let layout = Layout::array::<V>(cap).unwrap_or_else(|err| {
            unreachable!(
                "<{Self} as Drop>::drop: \
                 can't create array layout for type `{t}` of {cap} capacity: {err}",
                Self = type_name::<Self>(),
                t = type_name::<V>(),
                cap = cap,
                err = err,
            )
        });
        // SAFETY:
        // * `values` points to the object allocated with `layout`.
        // * `key_to_offset` is initialized.
        unsafe {
            dealloc(self.values.get_mut().as_ptr().cast::<u8>(), layout);
            self.key_to_offset.get_mut().assume_init_drop();
        }
    }
}

enum ReallocResult<K, V> {
    Success {
        keys: NonNull<K>,
        values: NonNull<V>,
        new_cap: NonZeroUsize,
    },
    KeysFailed {
        keys_layout: Layout,
    },
    ValuesFailed {
        keys: NonNull<K>,
        values_layout: Layout,
    },
}
