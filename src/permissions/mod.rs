//! Permission system for controlling tool execution authorization.
//!
//! Provides [`PermissionContext`] which encapsulates the full permission
//! checking logic: deny rules -> allow rules -> mode-based default.

pub mod checker;

pub use checker::PermissionContext;
