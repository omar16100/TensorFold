//! The real shared lane host dispatches the production prompt callbacks around history-dependent decode rounds.
const std = @import("std");
const lanes = @import("lanes");
const api = @import("../../core/engine_api.zig");
const Fake = @import("prefill_test_engine.zig").Fake;
const fill = @import("prefill.zig");
const a = std.testing.allocator;

const Box = struct {
    fake: *Fake,
    tokens: [2]std.ArrayList(u32) = @splat(.empty),
    prefills: [2]u32 = @splat(0),
    cached: [2]?u32 = @splat(null),
    reasons: [2]?api.Reason = @splat(null),
    first_order: [2]api.Id = undefined,
    n_first: usize = 0,
    done: std.atomic.Value(u32) = .init(0),
    bad: bool = false,
    fn event(ptr: *anyopaque, id: api.Id, e: *const api.Event) void {
        const b: *@This() = @ptrCast(@alignCast(ptr));
        switch (e.*) {
            .tokens => |ts| b.tokens[id - 1].appendSlice(a, ts) catch {
                b.bad = true;
            },
            .prefilled => |cached| {
                b.cached[id - 1] = cached;
                b.prefills[id - 1] += 1;
                if (b.n_first == b.first_order.len or b.fake.ends <= b.n_first) {
                    b.bad = true;
                } else {
                    b.first_order[b.n_first] = id;
                    b.n_first += 1;
                }
            },
            .finished => |end| {
                b.reasons[id - 1] = end.reason;
                if (end.reason == .failed) b.bad = true;
                _ = b.done.fetchAdd(1, .release);
            },
            else => {},
        }
    }
    fn deinit(b: *Box) void {
        for (&b.tokens) |*ts| ts.deinit(a);
    }
};
fn wait(flag: *const std.atomic.Value(bool)) !void {
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 30 * std.time.ns_per_s;
    while (!flag.load(.acquire)) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() > deadline) return error.TestTimeout;
        std.Thread.yield() catch {};
    }
}
fn finished(box: *const Box) !void {
    const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 30 * std.time.ns_per_s;
    while (box.done.load(.acquire) < 2) {
        if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() > deadline) return error.TestTimeout;
        std.Thread.yield() catch {};
    }
}
fn config() !lanes.Config {
    return lanes.Config.init(a, .{ .exact_width = 2, .mtp = true, .speculate = true, .drafts = 1, .hidden_rows = true, .streams_exact = true, .batch_rows = 4, .max_streams = 2, .draft_streams = true }, 2, 1);
}

test "flashnext prefill: an interrupted warm pass retains a cache mark and the next reply keeps its tokens" {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    const pc = api.prompt_cache;
    const Snapshots = struct {
        fn bytes(_: *anyopaque, at: u32) u64 {
            return 64 + 4 * @as(u64, at);
        }
        fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) !pc.Saved {
            const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            return target.save(@ptrCast(@alignCast(owner.?)), at);
        }
        fn restore(_: *anyopaque, _: ?*anyopaque, _: pc.Saved) !void {
            return error.BackendRestores;
        }
        fn drop(ptr: *anyopaque, saved: pc.Saved) void {
            const target: *lanes.fake.Fake = @ptrCast(@alignCast(ptr));
            target.drop(saved);
        }
    };
    const snapshots = pc.Snapshots{ .ptr = &f.target, .vtable = &.{ .bytes = Snapshots.bytes, .save = Snapshots.save, .restore = Snapshots.restore, .drop = Snapshots.drop } };
    var store = pc.Store.init(a, snapshots, .{ .lookahead = 1, .planned = true, .warm = .on, .min_prompt = 0, .min_gap = 1 }, 1 << 20);
    defer store.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2, .warm_turns = true });
    host.cache = &store;
    var box: Box = .{ .fake = &f };
    defer box.deinit();
    const prompt: [2051]u32 = @splat(5);
    const warm: api.Request = .{ .prompt = &prompt, .max_tokens = 0, .background = true, .history_len = prompt.len, .chunks = &.{ 1024, 2048 }, .shared_prefixes = &.{1024} };
    const real: api.Request = .{ .prompt = &prompt, .max_tokens = 16, .history_len = prompt.len, .chunks = &.{ 1024, 2048 }, .shared_prefixes = &.{1024} };
    const Arrival = struct {
        engine: api.Engine,
        request: *const api.Request,
        box: *Box,
        sent: bool = false,
        fn hook(ptr: *anyopaque, s: *lanes.Stream) void {
            const state: *@This() = @ptrCast(@alignCast(ptr));
            if (s.max_new != 0 or state.sent) return;
            state.sent = true;
            state.engine.submit(2, state.request, .{ .ctx = state.box, .event = Box.event }) catch @panic("test request failed");
        }
    };
    var arrival: Arrival = .{ .engine = host.engine(), .request = &real, .box = &box };
    f.chunk_hook = Arrival.hook;
    f.chunk_ctx = &arrival;
    try host.engine().submit(1, &warm, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    try finished(&box);
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqual(api.Reason.cancelled, box.reasons[0].?);
    try std.testing.expectEqual(api.Reason.length, box.reasons[1].?);
    try std.testing.expectEqual(@as(?u32, 1024), box.cached[1]);
    try std.testing.expectEqual(@as(usize, 16), box.tokens[1].items.len);
    var history: std.ArrayList(u32) = .empty;
    defer history.deinit(a);
    try history.appendSlice(a, &prompt);
    for (box.tokens[1].items) |token| {
        try std.testing.expectEqual(lanes.fake.next(history.items, null, history.items.len), token);
        try history.append(a, token);
    }
    try std.testing.expectEqual(f.owners, f.freed);
}

fn run(drafts: bool, sampled: bool, concurrent: bool) !Box {
    var f: Fake = .{ .barrier = concurrent };
    defer f.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    var box: Box = .{ .fake = &f };
    errdefer box.deinit();
    const prompt = try a.alloc(u32, 102400);
    defer a.free(prompt);
    for (prompt, 0..) |*token, i| token.* = @intCast((i * 3 + 7) % 97);
    const first: api.Request = .{ .prompt = &.{ 3, 1, 4, 1 }, .max_tokens = 8192, .drafts = drafts, .sampling = if (sampled) .{ .seed = 1235, .temperature = 1 } else null };
    const second: api.Request = .{ .prompt = prompt, .max_tokens = 16, .drafts = drafts, .sampling = if (sampled) .{ .seed = 1236, .temperature = 1 } else null };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer {
        f.go.store(true, .release);
        if (!stopped) host.stop();
    }
    if (concurrent) {
        try wait(&f.ready);
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
        f.go.store(true, .release);
    } else {
        const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 30 * std.time.ns_per_s;
        while (box.done.load(.acquire) < 1) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() > deadline) return error.TestTimeout;
            std.Thread.yield() catch {};
        }
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
    }
    try finished(&box);
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, &box.prefills);
    try std.testing.expectEqual(@as(usize, 8192), box.tokens[0].items.len);
    try std.testing.expectEqual(@as(usize, 16), box.tokens[1].items.len);
    try std.testing.expectEqual(@as(usize, 2), f.begins);
    try std.testing.expectEqual(@as(usize, 2), f.ends);
    try std.testing.expectEqual(f.owners, f.freed);
    if (concurrent) {
        try std.testing.expectEqual(@as(usize, 102400 / fill.decode_rows), f.busy_chunks);
        try std.testing.expect(f.max_busy_rows <= fill.decode_rows);
        try std.testing.expect(f.last_round - f.first_round >= f.busy_chunks - 1);
        try std.testing.expect(f.settled >= f.busy_chunks);
    }
    // The sink is finished; the returned token arrays have no borrowed engine or stream data.
    box.fake = undefined;
    return box;
}

test "flashnext prefill: a 100k fill yields to every decoding round with sampled and plain token equality" {
    for ([_]bool{ false, true }) |sampled| {
        var plain = try run(false, sampled, false);
        defer plain.deinit();
        var drafted = try run(true, sampled, false);
        defer drafted.deinit();
        var shared_plain = try run(false, sampled, true);
        defer shared_plain.deinit();
        var shared_drafted = try run(true, sampled, true);
        defer shared_drafted.deinit();
        for (0..2) |i| {
            try std.testing.expectEqualSlices(u32, plain.tokens[i].items, drafted.tokens[i].items);
            try std.testing.expectEqualSlices(u32, plain.tokens[i].items, shared_plain.tokens[i].items);
            try std.testing.expectEqualSlices(u32, plain.tokens[i].items, shared_drafted.tokens[i].items);
        }
    }
}

const Lone = struct {
    fake: *Fake,
    ready: std.atomic.Value(bool) = .init(false),
    go: std.atomic.Value(bool) = .init(false),
    partials: usize = 0,
    fn boundary(ptr: *anyopaque, _: *lanes.Stream) void {
        const l: *@This() = @ptrCast(@alignCast(ptr));
        if (!l.ready.load(.acquire)) {
            l.ready.store(true, .release);
            while (!l.go.load(.acquire)) std.Thread.yield() catch {};
        }
    }
    fn run(ptr: *anyopaque, s: *lanes.Stream, hooks: api.LoneHooks) anyerror!bool {
        const l: *@This() = @ptrCast(@alignCast(ptr));
        if (!try fill.lone(l.fake, s, hooks)) {
            l.partials += 1;
            return true;
        }
        defer l.fake.abortPrompt(s);
        const token = lanes.fake.next(s.prompt(), s.sampling, s.prompt_len);
        _ = try s.commit(a, &.{token}, &.{});
        s.pending = token;
        s.cache_len = s.prompt_len;
        hooks.committed(hooks.ctx);
        if (!s.finished) return error.ExpectedOneToken;
        return false;
    }
};

test "flashnext prefill: a long lone prompt hands unfinished work to the shared host before its first token" {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    var lone: Lone = .{ .fake = &f };
    f.chunk_hook = Lone.boundary;
    f.chunk_ctx = &lone;
    host.lone = .{ .ctx = &lone, .run = Lone.run };
    var box: Box = .{ .fake = &f };
    defer box.deinit();
    const prompt = try a.alloc(u32, 102400);
    defer a.free(prompt);
    @memset(prompt, 5);
    const first: api.Request = .{ .prompt = prompt, .max_tokens = 1 };
    const second: api.Request = .{ .prompt = &.{ 2, 7, 1 }, .max_tokens = 1 };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer {
        lone.go.store(true, .release);
        if (!stopped) host.stop();
    }
    try wait(&lone.ready);
    try std.testing.expectEqual(@as(usize, 0), box.n_first);
    try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
    lone.go.store(true, .release);
    try finished(&box);
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqual(@as(usize, 1), lone.partials);
    try std.testing.expectEqualSlices(api.Id, &.{ 2, 1 }, &box.first_order);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, &box.prefills);
    try std.testing.expectEqual(@as(usize, 2), f.begins);
    try std.testing.expectEqual(@as(usize, 2), f.ends);
    try std.testing.expectEqual(@as(usize, 1), box.tokens[0].items.len);
    try std.testing.expectEqual(@as(usize, 1), box.tokens[1].items.len);
    try std.testing.expectEqual(lanes.fake.next(prompt, first.sampling, prompt.len), box.tokens[0].items[0]);
    try std.testing.expectEqual(f.owners, f.freed);
}

fn longFirst(drafts: bool, sampled: bool, concurrent: bool) !Box {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try config();
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    var lone: Lone = .{ .fake = &f };
    if (concurrent) {
        f.chunk_hook = Lone.boundary;
        f.chunk_ctx = &lone;
        if (drafts) host.lone = .{ .ctx = &lone, .run = Lone.run };
    }
    var box: Box = .{ .fake = &f };
    errdefer box.deinit();
    const prompt = try a.alloc(u32, 102400);
    defer a.free(prompt);
    for (prompt, 0..) |*token, i| token.* = @intCast((i * 7 + 3) % 97);
    const first: api.Request = .{ .prompt = prompt, .max_tokens = 64, .drafts = drafts, .sampling = if (sampled) .{ .seed = 2027, .temperature = 1 } else null };
    const second: api.Request = .{ .prompt = &.{ 2, 7, 1 }, .max_tokens = 8192, .drafts = drafts, .sampling = if (sampled) .{ .seed = 2028, .temperature = 1 } else null };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer {
        lone.go.store(true, .release);
        if (!stopped) host.stop();
    }
    if (concurrent) {
        try wait(&lone.ready);
        try std.testing.expectEqual(@as(usize, 0), box.n_first);
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
        lone.go.store(true, .release);
    } else {
        const deadline = std.Io.Clock.awake.now(std.testing.io).toNanoseconds() + 30 * std.time.ns_per_s;
        while (box.done.load(.acquire) < 1) {
            if (std.Io.Clock.awake.now(std.testing.io).toNanoseconds() > deadline) return error.TestTimeout;
            std.Thread.yield() catch {};
        }
        try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
    }
    try finished(&box);
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, &box.prefills);
    try std.testing.expectEqual(@as(usize, 64), box.tokens[0].items.len);
    try std.testing.expectEqual(@as(usize, 8192), box.tokens[1].items.len);
    try std.testing.expectEqual(@as(usize, 2), f.begins);
    try std.testing.expectEqual(@as(usize, 2), f.ends);
    try std.testing.expectEqual(f.owners, f.freed);
    if (concurrent) {
        try std.testing.expectEqualSlices(api.Id, &.{ 2, 1 }, &box.first_order);
        try std.testing.expectEqual(@as(usize, if (drafts and !sampled) 1 else 0), lone.partials);
        try std.testing.expect(f.busy_chunks > 0);
        try std.testing.expect(f.max_busy_rows <= fill.decode_rows);
        try std.testing.expect(f.last_round - f.first_round >= f.busy_chunks - 1);
    }
    box.fake = undefined;
    return box;
}

test "flashnext prefill: long-first handoff keeps complete sampled and greedy drafted plain and solo replies" {
    for ([_]bool{ false, true }) |sampled| {
        var solo = try longFirst(false, sampled, false);
        defer solo.deinit();
        for ([_]bool{ false, true }) |drafts| {
            var concurrent = try longFirst(drafts, sampled, true);
            defer concurrent.deinit();
            for (0..2) |i| try std.testing.expectEqualSlices(u32, solo.tokens[i].items, concurrent.tokens[i].items);
        }
    }
}

fn shortRun(drafts: bool, sampled: bool, thinking: bool, batched: bool) !Box {
    var f: Fake = .{};
    defer f.deinit();
    var cfg = try lanes.Config.init(a, .{ .exact_width = 2, .mtp = true, .speculate = true, .drafts = 1, .hidden_rows = true, .streams_exact = true, .batch_rows = 4, .max_streams = 2, .draft_streams = batched }, 2, 1);
    defer cfg.deinit(a);
    var clock: lanes.fake.FixedClock = .{};
    var core = lanes.Engine.init(a, &cfg, f.backend(), clock.clock());
    defer core.deinit();
    var host = api.LaneHost.init(a, std.testing.io, &core, .{ .lanes = 2 });
    var box: Box = .{ .fake = &f };
    errdefer box.deinit();
    const first_prompt: [33]u32 = @splat(3);
    const second_prompt: [65]u32 = @splat(7);
    const first: api.Request = .{ .prompt = &first_prompt, .max_tokens = 64, .drafts = drafts, .sampling = if (sampled) .{ .seed = 53, .temperature = 1 } else null, .think_budget = if (thinking) 4 else 0, .think_close = &.{ 94, 95 }, .think_end = 95 };
    const second: api.Request = .{ .prompt = &second_prompt, .max_tokens = 64, .drafts = drafts, .sampling = if (sampled) .{ .seed = 79, .temperature = 1 } else null, .think_budget = if (thinking) 4 else 0, .think_close = &.{ 94, 95 }, .think_end = 95 };
    const e = host.engine();
    try e.submit(1, &first, .{ .ctx = &box, .event = Box.event });
    try e.submit(2, &second, .{ .ctx = &box, .event = Box.event });
    try host.start();
    var stopped = false;
    defer if (!stopped) host.stop();
    try finished(&box);
    host.stop();
    stopped = true;
    try std.testing.expect(!box.bad);
    try std.testing.expectEqualSlices(u32, &.{ 1, 1 }, &box.prefills);
    try std.testing.expectEqual(f.owners, f.freed);
    if (drafts) {
        if (batched) try std.testing.expect(f.target.drafts_most > 1) else try std.testing.expectEqual(@as(usize, 1), f.target.drafts_most);
    }
    box.fake = undefined;
    return box;
}

test "flashnext prefill: thinking and both shared head dispatch modes retain complete plain token IDs" {
    for ([_]bool{ false, true }) |sampled| for ([_]bool{ false, true }) |thinking| {
        var plain = try shortRun(false, sampled, thinking, false);
        defer plain.deinit();
        for ([_]bool{ false, true }) |batched| {
            var drafted = try shortRun(true, sampled, thinking, batched);
            defer drafted.deinit();
            for (0..2) |i| try std.testing.expectEqualSlices(u32, plain.tokens[i].items, drafted.tokens[i].items);
        }
    };
}
