/**
 * Multi-Provider — unified provider config and request/response transforms.
 *
 * Supports: Anthropic, OpenAI, Google, Bedrock (stub), Vertex (stub).
 * Each provider defines endpoint, auth headers, and optional transforms.
 */

const PROVIDERS = {
    anthropic: {
        name: 'Anthropic',
        endpoint: 'https://api.anthropic.com/v1/messages',
        envKey: 'ANTHROPIC_API_KEY',
        authHeader(key) {
            return {
                'x-api-key': key,
                'anthropic-version': '2023-06-01',
                'Content-Type': 'application/json',
            };
        },
        models: ['claude-sonnet-4-6', 'claude-haiku-4-5', 'claude-opus-4-6'],
    },

    openai: {
        name: 'OpenAI',
        endpoint: 'https://api.openai.com/v1/chat/completions',
        envKey: 'OPENAI_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['gpt-4o', 'gpt-4o-mini', 'o1-preview', 'o1-mini', 'o3-mini'],
        transformRequest(body) {
            const messages = [];
            if (body.system) {
                messages.push({ role: 'system', content: body.system });
            }
            for (const msg of body.messages || []) {
                if (typeof msg.content === 'string') {
                    messages.push({ role: msg.role, content: msg.content });
                } else if (Array.isArray(msg.content)) {
                    for (const block of msg.content) {
                        if (block.type === 'tool_result') {
                            messages.push({
                                role: 'tool',
                                tool_call_id: block.tool_use_id,
                                content: block.content,
                            });
                        }
                    }
                }
            }

            const tools = (body.tools || []).map(t => ({
                type: 'function',
                function: { name: t.name, description: t.description, parameters: t.input_schema },
            }));

            return {
                model: body.model,
                messages,
                ...(tools.length > 0 && { tools }),
                ...(body.max_tokens && { max_tokens: body.max_tokens }),
                ...(body.stream && { stream: true }),
            };
        },
        transformResponse(data) {
            const choice = data.choices?.[0];
            if (!choice) throw new Error('No choices in OpenAI response');

            const content = [];
            if (choice.message?.content) {
                content.push({ type: 'text', text: choice.message.content });
            }
            if (choice.message?.tool_calls) {
                for (const tc of choice.message.tool_calls) {
                    content.push({
                        type: 'tool_use',
                        id: tc.id,
                        name: tc.function.name,
                        input: JSON.parse(tc.function.arguments || '{}'),
                    });
                }
            }

            return {
                content,
                stop_reason: choice.finish_reason === 'stop' ? 'end_turn' : choice.finish_reason,
                usage: {
                    input_tokens: data.usage?.prompt_tokens || 0,
                    output_tokens: data.usage?.completion_tokens || 0,
                },
            };
        },
    },

    google: {
        name: 'Google',
        endpoint: 'https://generativelanguage.googleapis.com/v1beta/models',
        envKey: 'GOOGLE_API_KEY',
        altEnvKey: 'GEMINI_API_KEY',
        authHeader(key) {
            return { 'Content-Type': 'application/json' };
        },
        models: ['gemini-2.0-flash', 'gemini-2.0-pro', 'gemini-1.5-flash'],
        transformRequest(body) {
            const contents = [];
            for (const msg of body.messages || []) {
                const role = msg.role === 'assistant' ? 'model' : 'user';
                if (typeof msg.content === 'string') {
                    contents.push({ role, parts: [{ text: msg.content }] });
                }
            }

            return {
                contents,
                ...(body.system && {
                    systemInstruction: { parts: [{ text: body.system }] },
                }),
            };
        },
        transformResponse(data) {
            const candidate = data.candidates?.[0];
            if (!candidate) throw new Error('No candidates in Google response');

            const content = [];
            for (const part of candidate.content?.parts || []) {
                if (part.text) content.push({ type: 'text', text: part.text });
            }

            return {
                content,
                stop_reason: 'end_turn',
                usage: {
                    input_tokens: data.usageMetadata?.promptTokenCount || 0,
                    output_tokens: data.usageMetadata?.candidatesTokenCount || 0,
                },
            };
        },
    },

    bedrock: {
        name: 'AWS Bedrock',
        endpoint: null, // Dynamic based on region
        envKey: 'AWS_ACCESS_KEY_ID',
        models: ['anthropic.claude-3-sonnet', 'anthropic.claude-3-haiku'],
        authHeader() {
            // AWS SigV4 signing would go here
            return { 'Content-Type': 'application/json' };
        },
        getEndpoint(model, region = 'us-east-1') {
            return `https://bedrock-runtime.${region}.amazonaws.com/model/${model}/invoke`;
        },
    },

    vertex: {
        name: 'Google Vertex AI',
        endpoint: null, // Dynamic based on project/region
        envKey: 'GOOGLE_APPLICATION_CREDENTIALS',
        models: ['claude-sonnet-4-6@anthropic'],
        authHeader() {
            // GCP bearer token would go here
            return { 'Content-Type': 'application/json' };
        },
        getEndpoint(model, project, region = 'us-central1') {
            return `https://${region}-aiplatform.googleapis.com/v1/projects/${project}/locations/${region}/publishers/anthropic/models/${model}:rawPredict`;
        },
    },

    // Added in nightly sync v2.1.97
    azure: {
        name: 'Azure OpenAI',
        endpoint: null, // Dynamic based on resource
        envKey: 'AZURE_OPENAI_API_KEY',
        authHeader(key) {
            return {
                'api-key': key,
                'Content-Type': 'application/json',
            };
        },
        models: ['gpt-4o', 'gpt-35-turbo', 'gpt-4'],
        getEndpoint(model, resource, deployment) {
            return `https://${resource}.openai.azure.com/openai/deployments/${deployment}/chat/completions`;
        },
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    cohere: {
        name: 'Cohere',
        endpoint: 'https://api.cohere.ai/v1/chat',
        envKey: 'COHERE_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['command-r-plus', 'command-r', 'command'],
        // TODO: Add request/response transforms
    },

    // Added in nightly sync v2.1.97
    mistral: {
        name: 'Mistral AI',
        endpoint: 'https://api.mistral.ai/v1/chat/completions',
        envKey: 'MISTRAL_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['mistral-large', 'mistral-medium', 'mistral-small'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    perplexity: {
        name: 'Perplexity',
        endpoint: 'https://api.perplexity.ai/chat/completions',
        envKey: 'PERPLEXITY_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['llama-3.1-sonar-large-128k-online', 'llama-3.1-sonar-small-128k-online'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    together: {
        name: 'Together AI',
        endpoint: 'https://api.together.xyz/v1/chat/completions',
        envKey: 'TOGETHER_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['meta-llama/Llama-2-70b-chat-hf', 'togethercomputer/RedPajama-INCITE-Chat-3B-v1'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    fireworks: {
        name: 'Fireworks AI',
        endpoint: 'https://api.fireworks.ai/inference/v1/chat/completions',
        envKey: 'FIREWORKS_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['accounts/fireworks/models/llama-v3p1-70b-instruct', 'accounts/fireworks/models/mixtral-8x7b-instruct'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    groq: {
        name: 'Groq',
        endpoint: 'https://api.groq.com/openai/v1/chat/completions',
        envKey: 'GROQ_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['llama3-8b-8192', 'llama3-70b-8192', 'mixtral-8x7b-32768'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    huggingface: {
        name: 'Hugging Face',
        endpoint: 'https://api-inference.huggingface.co/models',
        envKey: 'HF_TOKEN',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['microsoft/DialoGPT-large', 'facebook/blenderbot-400M-distill'],
        getEndpoint(model) {
            return `https://api-inference.huggingface.co/models/${model}`;
        },
        // TODO: Add request/response transforms
    },

    // Added in nightly sync v2.1.97
    replicate: {
        name: 'Replicate',
        endpoint: 'https://api.replicate.com/v1/predictions',
        envKey: 'REPLICATE_API_TOKEN',
        authHeader(key) {
            return {
                'Authorization': `Token ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['meta/llama-2-70b-chat', 'stability-ai/stable-diffusion'],
        // TODO: Add request/response transforms
    },

    // Added in nightly sync v2.1.97
    anyscale: {
        name: 'Anyscale',
        endpoint: 'https://api.endpoints.anyscale.com/v1/chat/completions',
        envKey: 'ANYSCALE_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['meta-llama/Llama-2-70b-chat-hf', 'codellama/CodeLlama-34b-Instruct-hf'],
        // TODO: Add request/response transforms similar to OpenAI
    },

    // Added in nightly sync v2.1.97
    deepseek: {
        name: 'DeepSeek',
        endpoint: 'https://api.deepseek.com/v1/chat/completions',
        envKey: 'DEEPSEEK_API_KEY',
        authHeader(key) {
            return {
                'Authorization': `Bearer ${key}`,
                'Content-Type': 'application/json',
            };
        },
        models: ['deepseek-chat', 'deepseek-coder'],
        // TODO: Add request/response transforms similar to OpenAI
    },
};

/**
 * Get the provider configuration for a given model.
 * @param {string} model - model name
 * @returns {object} provider config
 */
export function getProvider(model) {
    if (model.startsWith('claude') || model.startsWith('anthropic')) return PROVIDERS.anthropic;
    if (model.startsWith('gpt') || model.startsWith('o1') || model.startsWith('o3')) return PROVIDERS.openai;
    if (model.startsWith('gemini')) return PROVIDERS.google;
    // Added in nightly sync v2.1.97 - extended model detection
    if (model.includes('azure')) return PROVIDERS.azure;
    if (model.startsWith('command')) return PROVIDERS.cohere;
    if (model.startsWith('mistral')) return PROVIDERS.mistral;
    if (model.includes('sonar')) return PROVIDERS.perplexity;
    if (model.includes('llama') || model.includes('RedPajama')) return PROVIDERS.together;
    if (model.includes('mixtral') || model.includes('fireworks')) return PROVIDERS.fireworks;
    if (model.includes('groq') || model.endsWith('-8192') || model.endsWith('-32768')) return PROVIDERS.groq;
    if (model.includes('/')) return PROVIDERS.huggingface; // HF model format
    if (model.startsWith('deepseek')) return PROVIDERS.deepseek;
    return PROVIDERS.anthropic; // default
}

/**
 * Get a provider by name.
 * @param {string} name
 * @returns {object|undefined}
 */
export function getProviderByName(name) {
    return PROVIDERS[name];
}

/**
 * List all supported providers.
 * @returns {Array<{ name: string, envKey: string, models: string[] }>}
 */
export function listProviders() {
    return Object.entries(PROVIDERS).map(([key, p]) => ({
        id: key,
        name: p.name,
        envKey: p.envKey,
        models: p.models || [],
        hasEndpoint: !!p.endpoint,
    }));
}

/**
 * Check which providers have API keys configured.
 * @returns {Array<{ id: string, name: string, configured: boolean }>}
 */
export function checkProviderKeys() {
    return Object.entries(PROVIDERS).map(([key, p]) => ({
        id: key,
        name: p.name,
        configured: !!(process.env[p.envKey] || (p.altEnvKey && process.env[p.altEnvKey])),
    }));
}

export { PROVIDERS };