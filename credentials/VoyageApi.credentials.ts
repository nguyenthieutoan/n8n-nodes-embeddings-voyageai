import type {
	ICredentialType,
	INodeProperties,
	ICredentialTestRequest,
} from 'n8n-workflow';

export class VoyageApi implements ICredentialType {
	name = 'voyageApi';
	displayName = 'Voyage AI API';
	documentationUrl = 'https://docs.voyageai.com/reference/embeddings-api';
	icon = 'file:voyage.svg' as const;
	properties: INodeProperties[] = [
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: {
				password: true,
			},
			default: '',
			required: true,
			description: 'API Key provided in Voyage AI dashboard.',
		},
		{
			displayName: 'Base URL Override',
			name: 'baseUrl',
			type: 'string',
			default: 'https://api.voyageai.com/v1',
			required: true,
			description: 'Base URL of the Voyage AI API or local proxy URL.',
		},
	];

	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.baseUrl || "https://api.voyageai.com/v1"}}',
			url: '/embeddings',
			method: 'POST',
			headers: {
				'Authorization': '=Bearer {{$credentials.apiKey}}',
				'Content-Type': 'application/json',
			},
			body: {
				input: ['n8n gateway verification testing'],
				model: 'voyage-4-lite',
			},
		},
	};
}
