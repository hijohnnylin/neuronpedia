import UmapProvider from '@/components/provider/umap-provider';
import List from './list';

export default async function Page(props: { params: Promise<{ id: string }> }) {
  const params = await props.params;
  return (
    <UmapProvider>
      <List listId={params.id} />
    </UmapProvider>
  );
}
