import { SimulationStudio } from "@/components/planning/simulation-studio";
export default async function SimulatePage({ params }: { params: Promise<{ id: string }> }) { const { id } = await params; return <SimulationStudio id={id} />; }
