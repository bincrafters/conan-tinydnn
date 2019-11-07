from conans import ConanFile, tools
import os

class TinydnnConan(ConanFile):
    name = "tiny-dnn"
    version = "1.0.0a3"
    url = "https://github.com/bincrafters/conan-tinydnn"
    description = "Header only, dependency-free deep learning framework in C++14."
    license = "https://github.com/tiny-dnn/tiny-dnn/blob/master/LICENSE"

    def source(self):
        source_url = "https://github.com/tiny-dnn/tiny-dnn"
        tools.get("{0}/archive/v{1}.tar.gz".format(source_url, self.version))
        extracted_dir = self.name + "-" + self.version
        os.rename(extracted_dir, "sources")
        #Rename to "sources" is a convention to simplify later steps

    def package(self):
        with tools.chdir("sources"):
            self.copy(pattern="LICENSE")
            self.copy(pattern="*.h", dst="include", src="sources/", keep_path=True)

    def package_id(self):
        self.info.header_only()
