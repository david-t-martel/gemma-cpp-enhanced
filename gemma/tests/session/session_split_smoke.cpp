// Copyright 2025 Gemma Project
// SPDX-License-Identifier: Apache-2.0
//
// Session subsystem smoke test (pre-refactor scaffold)
// Labels: session;refactor;smoke
// This will evolve as we split the monolithic session implementation.

#include <gtest/gtest.h>

#if defined(GEMMA_HAS_SESSION_MANAGEMENT)
#include "../../tools/session/Session.h"
#include "../../tools/session/SessionManager.h"
#endif

TEST(SessionRefactorSmoke, FeatureFlagPresentOrDisabledGraceful) {
#if defined(GEMMA_HAS_SESSION_MANAGEMENT)
    // Basic construction sanity
    using namespace gemma::session;
    Session s("test-smoke", 64);
    EXPECT_EQ(s.get_session_id(), "test-smoke");
#else
    GTEST_SKIP() << "Session subsystem disabled (GEMMA_ENABLE_SESSION=OFF)";
#endif
}
