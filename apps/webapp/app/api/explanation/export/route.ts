import { withOptionalUser } from '@/lib/with-user';
import { NextResponse } from 'next/server';

/**
@swagger
{
  "/api/explanation/export": {
    "get": {
      "tags": [
        "Explanations"
      ],
      "summary": "Export Explanations",
      "security": [{
          "apiKey": []
      }],
      "description": "Removed: Use https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/ instead."
    }
  }
}
*/
export const GET = withOptionalUser(async () => {
  return NextResponse.json(
    {
      newUrl: 'https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/',
      message:
        'This endpoint has been removed. Explanations exports can be downloaded more quickly and efficiently at https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/  Please contact support@neuronpedia.org if you have any questions.',
    },
    { status: 400 },
  );
});
