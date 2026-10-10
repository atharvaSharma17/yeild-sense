import { BatchDetail } from "@/components/batches/batch-detail";

export default async function BatchDetailPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = await params;
  return <BatchDetail id={id} />;
}
