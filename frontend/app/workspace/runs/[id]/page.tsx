import { OptimizationResults } from "@/components/planning/optimization-results";
export default async function RunPage({ params }: { params: Promise<{ id: string }> }) { const { id } = await params; return <OptimizationResults id={id} />; }
