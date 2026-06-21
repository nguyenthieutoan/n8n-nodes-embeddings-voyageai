jest.mock('@n8n/ai-utilities', () => ({
	logWrapper: jest.fn().mockImplementation((provider: any) => provider),
	getConnectionHintNoticeField: jest.fn().mockReturnValue({}),
}));

import { VoyageEmbeddingsModel } from '../nodes/Voyage/EmbeddingsVoyage.node';
import { NodeApiError, type ISupplyDataFunctions } from 'n8n-workflow';

describe('VoyageEmbeddingsModel Unit Tests', () => {
	let mockContext: any;

	beforeEach(() => {
		mockContext = {
			getNode: jest.fn().mockReturnValue({ name: 'Embeddings Voyage AI' }),
			helpers: {
				httpRequest: jest.fn(),
			},
		};
	});

	// 1. Happy Path Scenario
	test('Happy Path: should successfully generate embeddings and preserve input order', async () => {
		const mockResponse = {
			data: [
				{ index: 0, embedding: [0.1, 0.2, 0.3] },
				{ index: 1, embedding: [0.4, 0.5, 0.6] },
			],
		};
		mockContext.helpers.httpRequest.mockResolvedValue(mockResponse);

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
		}, mockContext);

		const result = await model.embedDocuments(['doc1', 'doc2']);

		expect(mockContext.helpers.httpRequest).toHaveBeenCalledWith({
			method: 'POST',
			url: 'https://api.voyageai.com/v1/embeddings',
			headers: {
				'Content-Type': 'application/json',
				'Authorization': 'Bearer test-api-key',
			},
			body: {
				input: ['doc1', 'doc2'],
				model: 'voyage-4',
				output_dtype: 'float',
			},
			json: true,
		});
		expect(result).toEqual([
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
		]);
	});

	// 2. Missing Data Scenario (Optional fields fallback)
	test('Missing Data: should apply defaults for missing optional fields and build payload correctly', async () => {
		mockContext.helpers.httpRequest.mockResolvedValue({
			data: [{ index: 0, embedding: [0.1, 0.2] }],
		});

		// Instantiate with minimal parameters
		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: '', // should fallback to default baseUrl
			modelName: 'voyage-4-lite',
		}, mockContext);

		await model.embedQuery('query text');

		expect(mockContext.helpers.httpRequest).toHaveBeenCalledWith(expect.objectContaining({
			url: 'https://api.voyageai.com/v1/embeddings',
			body: {
				input: ['query text'],
				model: 'voyage-4-lite',
				output_dtype: 'float',
			},
		}));
	});

	// 3. Scrambled Index Order Scenario (Duplicate Data / Out-of-order API sorting)
	test('Duplicate/Scrambled Data: should sort API returned elements by original index sequence', async () => {
		const mockResponse = {
			data: [
				{ index: 1, embedding: [0.4, 0.5, 0.6] },
				{ index: 0, embedding: [0.1, 0.2, 0.3] },
			],
		};
		mockContext.helpers.httpRequest.mockResolvedValue(mockResponse);

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
		}, mockContext);

		const result = await model.embedDocuments(['doc1', 'doc2']);

		// Result must be sorted by index: [index 0, index 1]
		expect(result).toEqual([
			[0.1, 0.2, 0.3],
			[0.4, 0.5, 0.6],
		]);
	});

	// 4. Failed API Scenario
	test('Failed API: should throw NodeApiError if http request fails', async () => {
		const apiError = new Error('Rate limit exceeded');
		mockContext.helpers.httpRequest.mockRejectedValue(apiError);

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
		}, mockContext);

		await expect(model.embedQuery('test')).rejects.toThrow(NodeApiError);
	});

	// 5. Empty Search Scenario
	test('Empty Search: should handle empty inputs and return empty arrays', async () => {
		mockContext.helpers.httpRequest.mockResolvedValue({ data: [] });

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
		}, mockContext);

		const result = await model.embedDocuments([]);
		expect(result).toEqual([]);
	});

	// Additional Check: Base64 decoding handles correctly
	test('Base64 Decoding (Float): should decode string embeddings when base64 encoding is specified', async () => {
		const floatArray = new Float32Array([0.1, 0.2]);
		const base64Str = Buffer.from(floatArray.buffer).toString('base64');

		mockContext.helpers.httpRequest.mockResolvedValue({
			data: [{ index: 0, embedding: base64Str }],
		});

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
			encodingFormat: 'base64',
			outputDtype: 'float',
		}, mockContext);

		const result = await model.embedDocuments(['doc1']);
		expect(result[0][0]).toBeCloseTo(0.1, 5);
		expect(result[0][1]).toBeCloseTo(0.2, 5);
	});

	test('Base64 Decoding (Int8): should decode string embeddings for int8 dtype', async () => {
		const intArray = new Int8Array([10, 20]);
		const base64Str = Buffer.from(intArray.buffer).toString('base64');

		mockContext.helpers.httpRequest.mockResolvedValue({
			data: [{ index: 0, embedding: base64Str }],
		});

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
			encodingFormat: 'base64',
			outputDtype: 'int8',
		}, mockContext);

		const result = await model.embedDocuments(['doc1']);
		expect(result[0]).toEqual([10, 20]);
	});

	test('Base64 Decoding (Uint8): should decode string embeddings for uint8/ubinary dtype', async () => {
		const uintArray = new Uint8Array([100, 200]);
		const base64Str = Buffer.from(uintArray.buffer).toString('base64');

		mockContext.helpers.httpRequest.mockResolvedValue({
			data: [{ index: 0, embedding: base64Str }],
		});

		const model = new VoyageEmbeddingsModel({
			apiKey: 'test-api-key',
			baseUrl: 'https://api.voyageai.com/v1',
			modelName: 'voyage-4',
			encodingFormat: 'base64',
			outputDtype: 'ubinary',
		}, mockContext);

		const result = await model.embedDocuments(['doc1']);
		expect(result[0]).toEqual([100, 200]);
	});

	// Test the EmbeddingsVoyage supplyData method itself
	test('EmbeddingsVoyage.supplyData: should execute successfully and return supplyData response wrapped in logWrapper', async () => {
		const { EmbeddingsVoyage } = require('../nodes/Voyage/EmbeddingsVoyage.node');
		const node = new EmbeddingsVoyage();
		const mockSupplyDataContext = {
			getNodeParameter: jest.fn().mockImplementation((paramName, index, fallback) => {
				if (paramName === 'model') return 'voyage-4';
				if (paramName === 'inputType') return 'query';
				if (paramName === 'options') return { outputDimension: 1024 };
				return fallback;
			}),
			getCredentials: jest.fn().mockResolvedValue({
				apiKey: 'test-key',
				baseUrl: 'https://api.voyageai.com/v1',
			}),
			getNode: jest.fn().mockReturnValue({ name: 'Embeddings Voyage AI' }),
			helpers: {
				httpRequest: jest.fn(),
			},
		} as any;

		const result = await node.supplyData.call(mockSupplyDataContext, 0);
		expect(result).toBeDefined();
		expect(result.response).toBeDefined();
	});
});
