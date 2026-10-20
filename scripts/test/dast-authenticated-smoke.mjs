import assert from "node:assert/strict";

// Run only against the disposable server created by dast-smoke.yml.
const baseUrl = process.argv[2];
const password = process.env.INITIAL_PASSWORD;
assert.ok(
  baseUrl && password,
  "A disposable server URL argument and INITIAL_PASSWORD are required"
);
const headers = { "Content-Type": "application/json", Origin: baseUrl };
async function request(path, method = "GET", body, expected = 200) {
  const response = await fetch(new URL(path, baseUrl), {
    method,
    headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    signal: AbortSignal.timeout(15_000),
    redirect: "error",
  });
  assert.equal(
    response.status,
    expected,
    `${method} ${path}: expected ${expected}, got ${response.status}`
  );
  return response;
}

await request("/api/keys/groups", "GET", undefined, 401);
const login = await request("/api/auth/login", "POST", { password });
const session = login.headers.getSetCookie().find((cookie) => cookie.startsWith("auth_token="));
assert.ok(session, "login must issue a session cookie");
headers.Cookie = session.split(";")[0];
const csrf = await (await request("/api/auth/csrf")).json();
assert.ok(csrf.token, "authenticated session must receive a CSRF token");
headers["x-omniroute-csrf"] = csrf.token;

let groupId;
try {
  const created = await (
    await request("/api/keys/groups", "POST", { name: `dast-smoke-${Date.now()}` }, 201)
  ).json();
  groupId = created.group?.id;
  assert.ok(groupId, "group creation must return an ID");
  const path = `/api/keys/groups/${encodeURIComponent(groupId)}`;
  await request(path);
  await request(`${path}/keys`);
  await request(`${path}/permissions`);
  const updated = await (await request(path, "PUT", { description: "authenticated smoke" })).json();
  assert.equal(updated.group.description, "authenticated smoke");
} finally {
  if (groupId) await request(`/api/keys/groups/${encodeURIComponent(groupId)}`, "DELETE");
}
await request("/api/auth/logout", "POST");
console.log("Authenticated management smoke passed (login, CSRF, group CRUD, logout).");
