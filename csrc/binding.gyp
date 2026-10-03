{
    "targets": [{
        "target_name": "tensor",
        "sources": ["./c_to_node.c","./tensor.c","./cpu.c","./MemoryPool.c","./wrap_to_class.c"],
        "conditions":[[
                "OS=='win'",{
                    "msvs_settings": {
                        "VCCLCompilerTool": {
                            "AdditionalOptions": [
                                '-utf-8'
                            ],
                        },
                    }
                }
            ]
        ]
    }],
    
}