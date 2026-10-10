import { RouteEditor } from "@/components/routes/route-editor";
export default async function RoutePage({ params }: { params: Promise<{ id: string }> }) { const { id } = await params; return <RouteEditor id={id} />; }
