import { NextRequest, NextResponse } from "next/server";

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const baseUrl = process.env.YIELDSENSE_API_BASE_URL;
  const apiKey = process.env.YIELDSENSE_API_KEY;

  if (!baseUrl || !apiKey) {
    return NextResponse.json({ error: "YieldSense API proxy is not configured." }, { status: 503 });
  }

  if (path[0] !== "v1" || path.some((segment) => !/^[a-zA-Z0-9._-]+$/.test(segment) || segment === "." || segment === "..")) {
    return NextResponse.json({ error: "Invalid API path." }, { status: 400 });
  }
  const origin = request.headers.get("origin");
  if (request.method !== "GET" && origin && origin !== request.nextUrl.origin) {
    return NextResponse.json({ error: "Cross-origin writes are not permitted." }, { status: 403 });
  }

  const target = new URL(`${path.join("/")}${request.nextUrl.search}`, `${baseUrl.replace(/\/$/, "")}/`);
  const headers = new Headers();
  headers.set("Authorization", `Bearer ${apiKey}`);

  for (const name of ["content-type", "if-match", "accept"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const method = request.method;
  let response: Response;
  try {
    response = await fetch(target, {
    method,
    headers,
    body: method === "GET" ? undefined : await request.arrayBuffer(),
    cache: "no-store",
    redirect: "manual",
    signal: AbortSignal.timeout(30_000),
    });
  } catch {
    return NextResponse.json({ error: "The backend could not be reached. Try again shortly." }, { status: 503 });
  }
  if (response.status >= 300 && response.status < 400) {
    return NextResponse.json({ error: "Unexpected backend redirect." }, { status: 502 });
  }

  const responseHeaders = new Headers();
  for (const name of ["content-type", "etag", "x-request-id"]) {
    const value = response.headers.get(name);
    if (value) responseHeaders.set(name, value);
  }

  return new NextResponse(response.body, { status: response.status, headers: responseHeaders });
}

export async function GET(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function POST(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}

export async function PUT(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  return proxy(request, context);
}
