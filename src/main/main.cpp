#include <iostream>

#include "file.h"
#include "logger.h"

#include "main/renderer.h"
#include "scene/importer.h"

#include "json.hpp"

NAMESPACE_BEGIN(krr)

// construct "reference" from "reference_dir"
bool constructReferenceFile(const std::string &renderConfigPath) {
	int maxDepth = -1;

	// [1] check if reference_dir & name exists
	auto config = gpContext->getGlobalConfig();
	fs::path referenceDir;
	std::string sceneName;

	if (config.contains("name")) {
		sceneName = config.value("name", "");
	} else {
		Log(Error, "[Reference Image] `scene_name` not specified in config!");
		return false;
	}

	if (config.contains("reference_dir")) {
		referenceDir = File::resolve(config.value("reference_dir", ""));
	} else {
		Log(Error, "[Reference Image] `reference_dir` not specified in config!");
		return false;
	}

	// [2] get max_depth from renderConfig
	const json renderConfig = File::loadJSON(renderConfigPath);
	if (renderConfig.contains("passes")) {
		for (const json &p : renderConfig["passes"]) {
			string name = p.at("name");
			RenderPass::SharedPtr pass{};
			if (p.contains("params")) {
				const json params = p["params"];
				if (params.contains("max_depth")) {
					maxDepth = params.value("max_depth", -1);
					Log(Success, "[Reference Image] Get Max Depth = %d, from %s", maxDepth,
						name.c_str());
				}
			}
		}
	}

	// [2] check dir exists
	referenceDir /= ("d" + std::to_string(maxDepth));
	if (!fs::exists(referenceDir)) {
		Log(Error, "[Reference Image] Reference directory %s does not exist!",
			referenceDir.string().c_str());
		return false;
	}

	// [3] check if reference file exists
	for (const auto &entry : fs::directory_iterator(referenceDir)) {
		const auto &filePath = entry.path();
		if (filePath.extension() == ".exr") {
			std::string fileName = filePath.stem().string();
			int length			 = fileName.length();
			// startwith
			int idx = fileName.find(sceneName);
			if (idx != 0) {
				continue;
			}
			// left is digit
			if (length > sceneName.length() && fileName[sceneName.length()] == '_') {
				// check if the rest is a digit
				std::string digitPart = fileName.substr(sceneName.length() + 1);
				if (std::all_of(digitPart.begin(), digitPart.end(), ::isdigit)) {
					// found a valid reference image
					Log(Info, "[Reference Image] Found reference image: %s",
						filePath.string().c_str());
					gpContext->updateGlobalConfig({{"reference", filePath.string()}});
					return true;
				}
			}
		}
	}

	// [4] if not found, log error
	Log(Error, "[Reference Image] No reference image found in %s!", referenceDir.string().c_str());
	return false;
}

extern "C" int main(int argc, char *argv[]) {
	gpContext = std::make_unique<Context>();
	fs::current_path(File::cwd());
	logInfo("Working directory: " + string(KRR_PROJECT_DIR));

	string sceneConfig	= "common/configs/nrrs/scenes/test.json";
	string renderConfig = "common/configs/nrrs/render/test.json";

	for (int i = 1; i < argc; i++) {
		if (string(argv[i]) == "-scene")
			sceneConfig = string(argv[++i]);
		else if (string(argv[i]) == "-method")
			renderConfig = string(argv[++i]);
	}

	RenderApp app;
	{ // load scene
		// app.loadConfigFrom(sceneConfig);
		json config = File::loadJSON(sceneConfig);
		if (config.find("scene_path") != config.end()) {
			json sceneConfigInner = File::loadJSON(config["scene_path"]);
			config.erase("scene_path");
			for (auto &it : sceneConfigInner.items()) {
				if (it.key() == "global") {
					config[it.key()].update(it.value(), true);
				} else {
					config[it.key()] = it.value();
				}
			}
		}
		app.loadConfig(config);
	}

	constructReferenceFile(renderConfig);

	app.loadConfigFrom(renderConfig);

	// set output directory, default is same as the config file directory.
	if (!File::outputDir().empty())
		File::setOutputDir(File::resolve("common/outputs") / fs::path(sceneConfig).stem() /
						   fs::path(renderConfig).stem());

	app.run();
	exit(EXIT_SUCCESS);
}

NAMESPACE_END(krr)