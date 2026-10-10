import { NextRequest, NextResponse } from "next/server";

async function proxy(request: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const baseUrl = process.env.YIELDSENSE_API_BASE_URL;
  const apiKey = process.env.YIELDSENSE_API_KEY;

  if (!baseUrl || !apiKey) {
    return NextResponse.json({ error: "YieldSense API proxy is not configured." }, { status: 503 });
  }

  const target = new URL(`${path.join("/")}${request.nextUrl.search}`, `${baseUrl.replace(/\/$/, "")}/`);
  const headers = new Headers();
  headers.set("Authorization", `Bearer ${apiKey}`);

  for (const name of ["content-type", "if-match", "accept"]) {
    const value = request.headers.get(name);
    if (value) headers.set(name, value);
  }

  const method = request.method;
  const response = await fetch(target, {
    method,
    headers,
    body: method === "GET" ? undefined : await request.arrayBuffer(),
    cache: "no-store",
  });

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
